import unittest, shutil, tempfile, subprocess
from pathlib import Path
import social_inventory as s


class RealVideoSimilarityTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg not available")
    def test_transcoded_video_matches_but_different_scene_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            fps=[]
            for i,(src,quality) in enumerate([("testsrc",2),("testsrc",8),("smptebars",2)]):
                path=Path(tmp)/f"{i}.mp4"
                subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i",f"{src}=size=160x120:rate=5","-t","4","-c:v","mpeg4","-q:v",str(quality),"-threads","1","-y",str(path)],check=True)
                fps.append(s.fingerprint_bytes(path.read_bytes(),"VIDEO",s.Budget()))
            self.assertNotEqual(fps[0]["sha256"],fps[1]["sha256"])
            self.assertIsNotNone(s.compare_fingerprints(fps[0],fps[1]))
            self.assertIsNone(s.compare_fingerprints(fps[0],fps[2]))
