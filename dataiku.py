
class Folder:
    """
    A mock of dataiku.Folder for local testing where
    no dataiku is installed
    """
    def get_download_stream(self, path: str):
        pass