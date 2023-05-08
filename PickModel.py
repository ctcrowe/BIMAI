import IndexNetwork as indexNetwork
import DetailTypeNetwork as detailTypeNetwork
import OLFNetwork as olfNetwork
import SpaceFunctionNetwork as sfNetwork
import VisNetwork as VisNetwork

def pick(text):
    if text == "Visibility":
        txt_path = "Dataset/VisibilityData.txt"
        path = "Models/VisibilityNetwork.pt"
        model = VisNetwork.VisModel()
        testMdl = VisNetwork.Text
        dataset = VisNetwork.VisDataset
        return txt_path, path, model, testMdl, datset
    if text == "Space Function":
        txt_path = "Datasets/SpaceFunctionNetworkData.txt"
        path = "Models/SpaceFunctionNetwork.pt"
        model = sfNetwork.SpaceFunctionModel()
        testMdl = sfNetwork.Test
        dataset = sfNetwork.SpaceFunctionDataset
        return txt_path, path, model, testMdl, dataset
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
    if text == "OLF":
        txt_path = "Datasets/OLFNetworkData.txt"
        path = "Models/OLFNetwork.pt"
        model = olfNetwork.OLFModel()
        testMdl = olfNetwork.Test
        dataset = olfNetwork.OLFDataset
        return txt_path, path, model, testMdl, dataset
    return None, None, None, None, None
