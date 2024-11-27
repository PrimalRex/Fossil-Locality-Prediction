import os

# HELPER FUNCTIONS FOR PFL
def createDirectoryIfNotExist(DIR):
    if DIR.exists():
        print(f"{DIR} Already Exists!")
    else:
        DIR.mkdir(exist_ok=True)

def getDirectoryFileCount(DIR):
    return len(getDirectoryFileNames(DIR))

def getDirectoryFileNames(DIR):
    return os.listdir(DIR)