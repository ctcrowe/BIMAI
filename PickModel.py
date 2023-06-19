import IndexNetwork as indexNetwork
import DetailTypeNetwork as detailTypeNetwork
import OLFNetwork as olfNetwork
import SpaceFunctionNetwork as sfNetwork
import VisNetwork as visNetwork
import ControlNetwork as control

def pick(text):
    if text == "Visibility":
        txt_path = "Datasets/VisibilityData.txt"
        path = "Models/VisibilityNetworkv2.pt"
        model = visNetwork.VisibilityModel()
        testMdl = visNetwork.Test
        dataset = visNetwork.VisibilityDataset
        return txt_path, path, model, testMdl, dataset
    if text == "Control":
        txt_path = "Datasets/ControlData.txt"
        path = "Models/ControlNetwork.pt"
        model = control.CtrlModel
        testMdl = control.Test
        dataset = control.CtrlDataset
        return txt_path, path, model, testMdl, dataset
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
