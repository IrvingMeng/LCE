from models.Classifier import Classifier, NormalizedClassifier
from models.ResNet import ResNet50
from models.trans import RBTBlock


def build_model(config, num_classes):
	# Build backbone
	print("Initializing model: {}".format(config.MODEL.NAME))
	if config.MODEL.NAME == 'resnet50':
		model = ResNet50(res4_stride=config.MODEL.RES4_STRIDE)
	else:
		raise KeyError("Invalid model: '{}'".format(config.MODEL.NAME))
	print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

	# Build classifier
	if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
		classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_classes)
	else:
		classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_classes)
	
	if config.LCE.USE_TRANS:
		in_planes = config.LCE.FEATDIM_OLD
		out_planes = configs.LCE.FEATDIM_NEW
		trans_forward = nn.Sequential(
			RBTBlock(in_planes, out_planes, num_paths),
			RBTBlock(in_planes, out_planes, num_paths),
			RBTBlock(in_planes, out_planes, num_paths),
			RBTBlock(in_planes, out_planes, num_paths))
		trans_backward = nn.Sequential(
			RBTBlock(out_planes, in_planes, num_paths),
			RBTBlock(out_planes, in_planes, num_paths),
			RBTBlock(out_planes, in_planes, num_paths),
			RBTBlock(out_planes, in_planes, num_paths))

		return model, classifier, trans_forward, trans_backward
	else:
		return model, classifier    