import IndexNetwork as indexNetwork
import DetailTypeNetwork as detailTypeNetwork

def pick(text):
    if text == "Index":
        txt_path = "Datasets/IndexNetworkData.txt"
        path = "Models/IndexNetwork.pt"
        model = indexNetwork.IndexModel()
        testMdl = indexNetwork.Test
        dataset = indexNetwork.IndexDataset
        return txt_path, path, model, testMdl, dataset
    if text == "Detail":
        txt_path = "Datasets/DetailTypeData.txt"
        path = "Models/DetailTypeNetwork.pt"
        model = detailTypeNetwork.DetailModel()
        testMdl = detailTypeNetwork.Test
        dataset = detailTypeNetwork.DetailTypeDataset
        return txt_path, path, model, testMdl, dataset
    return None, None, None, None, None