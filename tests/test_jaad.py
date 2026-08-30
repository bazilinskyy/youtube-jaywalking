"""Tests for direct JAAD XML parsing."""

import tempfile
import unittest
from pathlib import Path

from crowd_jaywalking.jaad import JAADDataset


ANNOTATION_XML = """<annotations>
<meta><task><size>3</size><original_size><width>100</width><height>50</height></original_size></task></meta>
<track label="pedestrian">
  <box frame="0" outside="0" occluded="0" xtl="10" ytl="10" xbr="30" ybr="40">
    <attribute name="id">0_1_1b</attribute><attribute name="old_id">pedestrian1</attribute>
    <attribute name="cross">not-crossing</attribute><attribute name="occlusion">none</attribute>
  </box>
  <box frame="1" outside="0" occluded="0" xtl="20" ytl="10" xbr="40" ybr="40">
    <attribute name="id">0_1_1b</attribute><attribute name="old_id">pedestrian1</attribute>
    <attribute name="cross">crossing</attribute><attribute name="occlusion">part</attribute>
  </box>
  <box frame="2" outside="0" occluded="0" xtl="30" ytl="10" xbr="50" ybr="40">
    <attribute name="id">0_1_1b</attribute><attribute name="old_id">pedestrian1</attribute>
    <attribute name="cross">crossing</attribute><attribute name="occlusion">none</attribute>
  </box>
</track>
</annotations>"""

ATTRIBUTES_XML = """<ped_attributes>
<pedestrian id="0_1_1b" old_id="pedestrian1" crossing="1" crossing_point="1"
 designated="ND" signalized="NS" />
</ped_attributes>"""

TRAFFIC_XML = """<traffic_scene><road_type>street</road_type>
<frame id="0" ped_crossing="0" ped_sign="0" stop_sign="0" traffic_light="n/a" />
<frame id="1" ped_crossing="1" ped_sign="0" stop_sign="0" traffic_light="green" />
<frame id="2" ped_crossing="1" ped_sign="0" stop_sign="0" traffic_light="green" />
</traffic_scene>"""


class JAADDatasetTests(unittest.TestCase):
    def test_loads_person_tracks_crossing_intervals_and_traffic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "annotations",
                "annotations_attributes",
                "annotations_traffic",
                "split_ids/default",
            ):
                (root / name).mkdir(parents=True, exist_ok=True)
            (root / "annotations/video_0001.xml").write_text(ANNOTATION_XML, encoding="utf-8")
            (root / "annotations_attributes/video_0001_attributes.xml").write_text(
                ATTRIBUTES_XML,
                encoding="utf-8",
            )
            (root / "annotations_traffic/video_0001_traffic.xml").write_text(
                TRAFFIC_XML,
                encoding="utf-8",
            )
            (root / "split_ids/default/train.txt").write_text("video_0001\n", encoding="utf-8")

            dataset = JAADDataset(root)
            self.assertEqual(dataset.video_ids("train"), ["video_0001"])
            video = dataset.load_video("video_0001")
            track = video.tracks["0_1_1b"]

        self.assertEqual((video.width, video.height, video.num_frames), (100, 50, 3))
        self.assertTrue(track.behaviour_annotated)
        self.assertTrue(track.is_crossing)
        self.assertEqual(track.crossing_intervals(), ((1, 2),))
        self.assertAlmostEqual(track.boxes[0].x1, 0.10)
        self.assertAlmostEqual(track.boxes[0].y2, 0.80)
        self.assertEqual(track.occlusion[1], 1)
        self.assertEqual(track.attributes["designated"], "ND")
        self.assertEqual(video.traffic[1]["ped_crossing"], "1")
        self.assertEqual(video.road_type, "street")


if __name__ == "__main__":
    unittest.main()
