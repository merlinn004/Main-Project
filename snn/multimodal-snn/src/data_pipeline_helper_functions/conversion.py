from pathlib import Path
import shutil
import subprocess

def cnv2mp3(src_folder, dst_folder, overwrite=False):
    src = Path(src_folder)
    dst = Path(dst_folder)
    dst.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        raise FileNotFoundError("ffmpeg not found on PATH. Install ffmpeg or add it to PATH.")

    for mp4 in sorted(src.glob("*.mp4")):
        out = dst / (mp4.stem + ".wav")
        if out.exists() and not overwrite:
            print("skip:", out.name)
            continue

        cmd = [
            ffmpeg_exe,
            "-y" if overwrite else "-n",
            "-i",
            str(mp4),
            "-vn",
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "2",
            str(out),
        ]

        # run without check=True so we can inspect output and handle failures
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip()
            print(f"ffmpeg failed for {mp4.name} (rc={proc.returncode}). Reason (first lines):")
            for line in (err.splitlines()[:6]):
                print("  ", line)

            # delete unreadable source and any partial target
            try:
                if mp4.exists():
                    mp4.unlink()
                    print("deleted video:", mp4.name)
            except Exception as e:
                print("failed to delete video:", mp4.name, e)

            try:
                if out.exists():
                    out.unlink()
                    print("deleted audio:", out.name)
            except Exception as e:
                print("failed to delete audio:", out.name, e)

            continue

        print("wrote:", out.name)
