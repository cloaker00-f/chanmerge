import os
import sys
import glob
import shutil
import subprocess
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.heasarc import Heasarc

def auto_merge_obs(ra=None, dec=None, radius_arcmin=None, energy_band="broad", outdir=None):
    """
    ====================================================================
    Automated Chandra X-ray Data Pipeline
    ====================================================================
    Description:
        Queries HEASARC for Chandra observations within a specified radius 
        of a target RA/Dec, downloads the primary data, reprocesses it 
        with the latest calibrations, and merges it into a flux image.

    Required Parameters:
        ra (str)            : Right Ascension. MUST be a sexagesimal string (e.g., "13:25:27.6") 
                              or explicitly include the degree unit if using decimals (e.g., "201.342d").
                              WARNING: Raw floats/strings without units (e.g., 201.342) will be parsed as hours!
        dec (str)           : Declination. MUST be a sexagesimal string (e.g., "-43:01:09")
                              or explicitly include the degree unit if using decimals (e.g., "-43.0192d").
        radius_arcmin (float): Search radius (arcminutes)

    Optional Parameters:
        energy_band (str)   : Energy band(s) to be extracted and merged. Default: "broad".
                              Available options and usage:
                              - Single band: "broad", "soft", "medium", or "hard".
                              - Multiple bands: Comma-separated without spaces (e.g., "soft,hard").
                              - All standard bands: Use "csc" to automatically generate
                                separate images for broad, soft, medium, and hard bands.
        outdir (str)        : Directory to save all downloaded and processed files. Default: obs_{ra}_{dec}.
        

    Usage Example:
        from chanmerge import auto_merge_obs
        
        auto_merge_obs(
            ra="13:25:27.6", 
            dec="-43:01:09", 
            radius_arcmin=15.0
        )
    ====================================================================
    """
    
    # ==========================================
    # INPUT VALIDATION & AUTOMATIC HELP TRIGGER
    # ==========================================
    # 1. Check for missing required inputs
    if ra is None or dec is None or radius_arcmin is None:
        print("\n[!] ERROR: Missing required parameters.")
        print("Please review the function documentation below:\n")
        print(auto_merge_obs.__doc__)
        return

    # 2. Type validation for radius
    try:
        radius_arcmin = float(radius_arcmin)
    except (ValueError, TypeError):
        print("\n[!] ERROR: 'radius_arcmin' must be a numeric value (e.g., 15 or 15.0).")
        print("Please review the function documentation below:\n")
        print(auto_merge_obs.__doc__)
        return

    # ==========================================
    # PIPELINE EXECUTION
    # ==========================================
    print(f"\n--- Starting Chandra Pipeline for [{ra}, {dec}] ---")

    safe_ra = str(ra).replace(":", "")
    safe_dec = str(dec).replace(":", "")
    auto_dirname = f"obs_{safe_ra}_{safe_dec}"

    if outdir is None:
        outdir = auto_dirname
    else:
        original_input = outdir 
        outdir = str(outdir).replace(":", "-").replace("/", "_").replace("\\", "_").replace(" ", "_")
        outdir = outdir.lstrip('.')
        
        if not outdir:
            outdir = auto_dirname
            print(f"\n [!] WARNING: Your requested directory name '{original_input}' was invalid or entirely stripped for safety.")
            print(f"     Falling back to auto-generated name: '{outdir}'\n")
    
    print("\n[1/4] Querying HEASARC database...")
    heasarc = Heasarc()
    try:
        target_coordinate = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
    except Exception as e:
        print(f"\n[!] ERROR: Invalid coordinate format. Details: {e}")
        return

    table = heasarc.query_region(target_coordinate, catalog='chanmaster', radius=radius_arcmin * u.arcmin)
    
    standard_obs = []
    grating_obs = []

    if 'grating' in table.colnames:
        for row in table:
            obs_str = str(row['obsid']).strip()
            grat_val = row['grating']
            
            grat_str = grat_val.decode('utf-8').strip() if isinstance(grat_val, bytes) else str(grat_val).strip()
            
            if grat_str.upper() == 'NONE':
                standard_obs.append(obs_str)
            else:
                grating_obs.append(obs_str)
    else:
        standard_obs = [str(obs).strip() for obs in table['obsid']]

    print(f"Search complete. Found {len(standard_obs)} standard and {len(grating_obs)} grating observations.")

    if len(standard_obs) == 0 and len(grating_obs) == 0:
        print("No observations found in this region. Exiting pipeline.")
        return
    
    # ==========================================
    # INTERACTIVE PROMPTS FOR GRATING OBS
    # ==========================================
    download_grating = False
    repro_grating = False

    if grating_obs:
        print("\n--- [GRATING OBSERVATIONS DETECTED] ---")
        print("  [!] NOTE: Grating observations MUST be reprocessed to be masked")
        print("            and included in the final merged image. If you skip reprocessing,")
        print("            they will be kept as raw archive data and excluded from merging.")
        
        ans_down = input(f"> Found {len(grating_obs)} grating (HETG/LETG) observation(s). Do you want to download them? (y/n): ")
        if ans_down.lower() == 'y':
            download_grating = True

            ans_repro = input("> Do you want to run 'chandra_repro' on these grating observations? (y/n): ")
            if ans_repro.lower() == 'y':
                repro_grating = True
        else:
            print("  -> Grating observations will be excluded completely.")
            grating_obs = []
    
    
    # ==========================================
    # SANITY CHECK & STORAGE ESTIMATOR
    # ==========================================
    total_obs = len(standard_obs) + len(grating_obs)

    if total_obs == 0:
        print("No observations to download. Exiting pipeline.")
        return 
        
    try:
        valid_obsids = standard_obs + grating_obs
        total_exposure_sec = 0.0
        
        for row in table:
            obs_str = str(row['obsid']).strip()
            if obs_str in valid_obsids:
                try:
                    total_exposure_sec += float(row['exposure'])
                except (ValueError, TypeError):
                    pass 
        
        # Raw Data: ~50 MB fixed overhead per observation + ~2 MB per kilosecond (1000s) of exposure
        estimated_raw_mb = (total_obs * 50.0) + ((total_exposure_sec / 1000.0) * 2.0)
        
        # Processed Footprint: Repro (level 2 events, bad pixel maps) + Merged files (exposure maps)
        # typically take about 3.5x to 4x the space of the raw data.
        estimated_total_mb = estimated_raw_mb * 3.5

        def format_size(size_in_mb):
            return f"{size_in_mb / 1024:.2f} GB" if size_in_mb >= 1000 else f"{size_in_mb:.0f} MB"
            
        print("\n--- [STORAGE FOOTPRINT ESTIMATE] ---")
        print(f" Total Exposure Time : {total_exposure_sec / 1000:.1f} kiloseconds")
        print(f" Estimated Download  : ~{format_size(estimated_raw_mb)}")
        print(f" Final Disk Footprint: ~{format_size(estimated_total_mb)} (after reprocessing & merging)")
        print("------------------------------------\n")
        
    except KeyError:
        # Fallback in case the 'exposure' column is missing from the HEASARC response
        print("\n--- [STORAGE FOOTPRINT ESTIMATE] ---")
        print(f" Estimated Download  : ~{total_obs * 100} MB")
        print(f" Final Disk Footprint: ~{total_obs * 350} MB")
        print("\n [!] WARNING: These are rough estimates based on average observation sizes.")
        print(" [!] It is HIGHLY RECOMMENDED to have at least TWICE the estimated disk space")
        print("     available to account for temporary files and processing overhead.")
        print("------------------------------------\n")

    user_input = input(f"> Do you want to proceed with {total_obs} observations? (y/n): ")
    if user_input.lower() != 'y':
        print("Pipeline aborted by the user. No files were downloaded.")
        return

    print(f"\n[0/4] Preparing output directory: '{outdir}'...")
    os.makedirs(outdir, exist_ok=True)
        
    print("\n[2/4] Downloading primary data...")
    
    for obsid in standard_obs:
        print(f"  -> Downloading Standard ObsID: {obsid}")
        subprocess.run(f"download_chandra_obsid {obsid}", shell=True, check=True, cwd=outdir)

    if download_grating:
        for obsid in grating_obs:
            print(f"  -> Downloading Grating ObsID: {obsid}")
            subprocess.run(f"download_chandra_obsid {obsid}", shell=True, check=True, cwd=outdir)
            
            old_path = os.path.join(outdir, obsid)
            new_path = os.path.join(outdir, f"{obsid}_g")
            
            if os.path.exists(old_path):
                if os.path.exists(new_path):
                    shutil.rmtree(new_path)
                os.rename(old_path, new_path)
            else:
                print(f"  [!] WARNING: ObsID {obsid} download failed or skipped. Renaming aborted.")
        
    print("\n[3/4] Reprocessing data with latest calibrations...")
    
    for obsid in standard_obs:
        print(f"  -> Reprocessing Standard ObsID: {obsid}")
        subprocess.run(f"chandra_repro indir={obsid} outdir={obsid}/repro", shell=True, check=True, cwd=outdir)
        subprocess.run("punlearn ardlib", shell=True, check=True, cwd=outdir)

    if repro_grating:
        for obsid in grating_obs:
            print(f"  -> Reprocessing Grating ObsID: {obsid}_g")
            subprocess.run(f"chandra_repro indir={obsid}_g outdir={obsid}_g/repro", shell=True, check=True, cwd=outdir)
            subprocess.run("punlearn ardlib", shell=True, check=True, cwd=outdir)

    # ==========================================
    # MASKING PHASE (Grating Obs Only)
    # ==========================================
   
    processed_grating_files = []
    
    if repro_grating:
        print("\n[3.5/4] Masking diffraction arms from grating observations...")
        for obsid in grating_obs:
            evt2_pattern = os.path.join(outdir, f"{obsid}_g", "repro", "*_evt2.fits")
            evt_files = glob.glob(evt2_pattern)
            
            if evt_files:
                g_file = evt_files[0] 
                mask_file = g_file.replace("_evt2.fits", "_mask.fits")
                filtered_file = g_file.replace("_evt2.fits", "_filtered_evt2.fits")
                
                print(f"  -> Masking ObsID: {obsid}_g")
                
                # Create the mask
                subprocess.run(["tg_create_mask", f"infile={g_file}", f"outfile={mask_file}"], check=True)
                
                # Filter syntax
                filter_expression = f"{g_file}[exclude sky=region({mask_file})]"
                
                subprocess.run(["dmcopy", filter_expression, filtered_file], check=True)
                
                processed_grating_files.append(filtered_file)
            else:
                print(f"  [!] WARNING: Reprocessed evt2 file not found for {obsid}_g. Skipping mask.")
        
    print("\n[4/4] Merging event files into a single flux image...")
    event_files_list = []
    
    # Add standard observations
    for obsid in standard_obs:
        evt_path = glob.glob(os.path.join(outdir, obsid, "repro", "*_evt2.fits"))
        event_files_list.extend(evt_path)

    # Add masked grating observations (if empty, nothing is added)
    event_files_list.extend(processed_grating_files)

    if not event_files_list:
        print("\n[!] ERROR: No event files found for merging. Merging failed.")
        return
        
    total_merged = len(event_files_list)
    gratings_merged = len(processed_grating_files)
    standards_merged = total_merged - gratings_merged
    
    print(f"  -> Found {total_merged} event files for merging.")
    print(f"     ({standards_merged} standard, {gratings_merged} masked grating)")
        
    files_to_merge = ",".join(event_files_list)
    merged_dir = os.path.join(outdir, "merged_final_image")
    os.makedirs(merged_dir, exist_ok=True)
    outroot = os.path.join(merged_dir, "merged_obs")
    
    merge_command = f"merge_obs {files_to_merge} {outroot} bands={energy_band}"
    subprocess.run(merge_command, shell=True, check=True)
    
    print(f"\n--- Pipeline Completed Successfully! ---")
    print(f"All raw data, repro folders, and final merged products are inside '{outdir}'.")