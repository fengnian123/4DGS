import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from mcap.reader import make_reader
from mcap.writer import Writer
from mcap_ros2.decoder import DecoderFactory

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    selection = json.loads((root / 'clip/selection.json').read_text())
    source = Path(selection['source_mcap'])
    begin = selection['start_capture_ns'] - 250_000_000
    end = selection['start_capture_ns'] + int(selection['duration_s'] * 1e9) + 250_000_000
    counts, clocks, schemas, channels = Counter(), {}, {}, {}
    destination = root / 'clip/selected.mcap'
    with source.open('rb') as stream, destination.open('wb') as output:
        reader = make_reader(stream)
        writer = Writer(output)
        writer.start(profile=reader.get_header().profile)
        for attachment in reader.iter_attachments():
            writer.add_attachment(attachment.create_time, attachment.log_time, attachment.name, attachment.media_type, attachment.data)
        for metadata in reader.iter_metadata():
            writer.add_metadata(metadata.name, metadata.metadata)
        factory = DecoderFactory()
        for schema, channel, message in reader.iter_messages():
            stamp, basis = message.log_time, 'log_time'
            decoder = factory.decoder_for(channel.message_encoding, schema) if schema is not None and '::' not in schema.name else None
            if decoder is not None:
                decoded = decoder(message.data)
                timestamp = getattr(decoded, 'timestamp', None)
                if timestamp is None and hasattr(decoded, 'header'):
                    timestamp = decoded.header.stamp
                if timestamp is not None and hasattr(timestamp, 'sec'):
                    stamp = timestamp.sec * 10**9 + timestamp.nanosec
                    basis = 'capture_timestamp' if hasattr(decoded, 'timestamp') else 'header_stamp'
            if not begin <= stamp < end:
                continue
            schema_id = 0
            if schema is not None:
                if schema.id not in schemas:
                    schemas[schema.id] = writer.register_schema(schema.name, schema.encoding, schema.data)
                schema_id = schemas[schema.id]
            if channel.id not in channels:
                channels[channel.id] = writer.register_channel(channel.topic, channel.message_encoding, schema_id, channel.metadata)
            writer.add_message(channels[channel.id], message.log_time, message.data, message.publish_time, message.sequence)
            counts[channel.topic] += 1
            clocks[channel.topic] = basis
        writer.add_metadata('agentos_dynamic_selection', {'source': str(source), 'begin_ns': str(begin), 'end_ns': str(end), 'padding_seconds': '.25'})
        writer.finish()
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    with destination.open('rb') as stream:
        summary = make_reader(stream).get_summary()
        assert summary.statistics.message_count == sum(counts.values())
    manifest = {'source_mcap': str(source), 'source_sha256': digest, 'selected_mcap': str(destination), 'begin_ns': begin, 'end_ns': end, 'padding_seconds': .25, 'message_counts': dict(counts), 'selection_clock_by_topic': clocks, 'payloads': 'Original message bytes, sequence, log time and publish time preserved; includes calibration attachments and source metadata.'}
    (root / 'clip/raw_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
