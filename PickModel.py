import IndexNetwork as indexNetwork

def pick(text):
    if text == "Index":
        txt_path = "Datasets/IndexNetworkData.txt"
        path = "Models/IndexNetwork.pt"
        model = indexNetwork.IndexModel()
        testMdl = indexNetwork.Test
        return txt_path, path, model, testMdl
    return None, None, None, None