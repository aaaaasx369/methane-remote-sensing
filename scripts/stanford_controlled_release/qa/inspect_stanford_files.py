from pathlib import Path
import xml.etree.ElementTree as ET

xml_path = Path("metadata.xml")

tree = ET.parse(xml_path)
root = tree.getroot()

files = []

for elem in root.iter():
    tag = elem.tag.split("}")[-1].lower()

    if tag == "file":
        info = dict(elem.attrib)

        filename = (
            info.get("id")
            or info.get("name")
            or info.get("filename")
            or info.get("path")
        )

        if filename:
            files.append((filename, info))

print("=" * 80)
print("STANFORD SDR FILE INVENTORY")
print("=" * 80)

print("Number of file entries:", len(files))
print()

for i, (filename, info) in enumerate(files, 1):
    print(f"[{i}] {filename}")
    for k, v in info.items():
        if k != "id":
            print(f"    {k}: {v}")

print()
print("=" * 80)

Path("stanford_file_list.txt").write_text(
    "\n".join(x[0] for x in files),
    encoding="utf-8"
)

print("Saved: stanford_file_list.txt")
