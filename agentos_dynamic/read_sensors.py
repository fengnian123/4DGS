import json
from pathlib import Path

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

ROOT = Path(__file__).resolve().parent
path = Path(json.loads((ROOT / "inventory.json").read_text())["files"][0]["file"])
with path.open("rb") as stream:
    reader = make_reader(stream)
    for attachment in reader.iter_attachments():
        if attachment.name == "calibration/calib_para.yml":
            (ROOT / "calib_para.yml").write_bytes(attachment.data)
            print(attachment.data.decode())
    factory = DecoderFactory()
    seen = set()
    for schema, channel, message in reader.iter_messages(topics=["/sensor/camera/front_fisheye_encoding", "/LIO_ODOM_HIGH_FREQUENCY", "/sensor/lidar/front_lidar_pcd", "/sensor/lidar/rear_lidar_pcd"]):
        if channel.topic in seen:
            continue
        decoded = factory.decoder_for(channel.message_encoding, schema)(message.data)
        print("TOPIC", channel.topic, "log/publish", message.log_time, message.publish_time)
        if "camera" in channel.topic:
            print("timestamp", decoded.timestamp.sec, decoded.timestamp.nanosec, "frame", decoded.frame_id, "format", decoded.format, "bytes", len(decoded.data))
        elif "ODOM" in channel.topic:
            print("header", decoded.header.stamp.sec, decoded.header.stamp.nanosec, decoded.header.frame_id, "child", decoded.child_frame_id)
            print("pose", decoded.pose.pose, "twist", decoded.twist.twist)
        else:
            print("header", decoded.header.stamp.sec, decoded.header.stamp.nanosec, decoded.header.frame_id, "dimensions", decoded.width, decoded.height, "fields", decoded.fields)
        seen.add(channel.topic)
        if len(seen) == 4:
            break
