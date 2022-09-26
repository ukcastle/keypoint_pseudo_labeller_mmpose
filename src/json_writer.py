from pathlib import Path
import json
import shutil
class JsonWriter:

  def __init__(self, path) -> None:
    self.filePath = Path(path)
    self.outputDict = {}
    self.loadFile()
  
  def loadFile(self):
    if self.filePath.is_file():
      with self.filePath.open("r") as f:
        self.outputDict = json.load(f)  

  def saveFile(self):
    with self.filePath.open("w") as f:
      json.dump(self.outputDict, f, indent=2)
