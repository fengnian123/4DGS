import argparse
import json
from pathlib import Path

from mcap.reader import make_reader

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
SOURCE = parser.parse_args().source.expanduser().resolve()
ROOT = Path(__file__).resolve().parent
records = []
for path in sorted(SOURCE.rglob("*.mcap")):
    with path.open("rb") as stream:
        summary = make_reader(stream).get_summary()
        statistics = summary.statistics
        channels = []
        for channel_id, channel in summary.channels.items():
            schema = summary.schemas.get(channel.schema_id)
            channels.append({"topic": channel.topic, "type": schema.name if schema else None, "encoding": channel.message_encoding, "count": statistics.channel_message_counts.get(channel_id, 0)})
        records.append({"file": str(path), "bytes": path.stat().st_size, "start_ns": statistics.message_start_time, "end_ns": statistics.message_end_time, "duration_s": (statistics.message_end_time - statistics.message_start_time) / 1e9, "channels": channels})
report = {"source": str(SOURCE), "files": records}
(ROOT / "inventory.json").write_text(json.dumps(report, indent=2))
print(f"{len(records)} MCAP files, {sum(record['bytes'] for record in records) / 1e9:.2f} GB")
for record in records:
    print(Path(record["file"]).name, round(record["duration_s"], 2), {channel["topic"]: channel["count"] for channel in record["channels"]})
