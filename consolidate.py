import argparse
import glob
import os
import pickle
from loguru import logger
import numpy as np
from ensemble_boxes import *

def main(args):
	# find in the input directory all the pickle files
	files = glob.glob(os.path.join(args.input[0], '*.pkl'))
	# for each pickle file
	for file in files:
		uid = os.path.basename(file).split('_')[0]
		predictions_boxes = pickle.load(open(file, 'rb'))['pred_boxes']
		predictions_scores = pickle.load(open(file, 'rb'))['pred_scores']
		predictions_labels = pickle.load(open(file, 'rb'))['pred_labels']
		spacing = pickle.load(open(file, 'rb'))['itk_spacing']
		origin = pickle.load(open(file, 'rb'))['itk_origin']
		direction = pickle.load(open(file, 'rb'))['itk_direction']
		size = pickle.load(open(file, 'rb'))['original_size_of_raw_data']
		name = os.path.basename(file)
		# find in other folders
		for infile in args.input[1:]:
			if os.path.isfile(os.path.join(infile, name)):
				# if the file is found, load it
				predictions_boxes = np.concatenate((predictions_boxes, pickle.load(open(os.path.join(infile, name), 'rb'))['pred_boxes']))
				predictions_scores = np.concatenate((predictions_scores, pickle.load(open(os.path.join(infile, name), 'rb'))['pred_scores']))
				predictions_labels = np.concatenate((predictions_labels, pickle.load(open(os.path.join(infile, name), 'rb'))['pred_labels']))
			else:
				logger.info(f'File {name} not found in {infile}')

		# top k=100
		idx = np.argsort(predictions_scores)[::-1][:100]
		idx = idx[:100]
		all_boxes = predictions_boxes[idx]
		all_scores = predictions_scores[idx]
		all_labels = predictions_labels[idx]

		# apply wbs
		iou_thr = 0.005
		skip_box_thr = 0.0001
		sigma = 0.1
		weights = [1]

		# all_boxes_new = []
		# scale to 0-1
		all_boxes_new = [
			[b[4] / size[0],
			 b[1] / size[1],
			 b[0] / size[2],
			 b[5] / size[0],
			 b[3] / size[1],
			 b[2] / size[2]] for b in
			list(all_boxes)] # x1, y1, z1, x2, y2, z2
		all_boxes_new, all_scores_new, all_labels_new = weighted_boxes_fusion_3d([all_boxes_new], [list(all_scores)],
																				 [list(all_labels)], weights=weights,
																				 iou_thr=iou_thr,
																				 skip_box_thr=skip_box_thr)

		# scale back to original size
		all_boxes_new = [
			[b[2] * size[2], b[1] * size[1], b[5] * size[2], b[4] * size[1], b[0] * size[0], b[3] * size[0]] for b in
			list(all_boxes_new)]
		all_boxes_new = np.array(all_boxes_new)
		all_scores_new = np.array(all_scores_new)
		all_labels_new = np.array(all_labels_new)

		result = {
			"pred_boxes": all_boxes_new,
			"pred_scores": all_scores_new,
			"pred_labels": all_labels_new,
			"itk_spacing": spacing,
			"itk_origin": origin,
			"itk_direction": direction,
			"original_size_of_raw_data": size,
			"restore": True
		}

		# save result
		save_path = os.path.join(args.output, uid + "_boxes.pkl")
		with open(save_path, "wb") as f:
			pickle.dump(result, f)
		print("Saved result to {}".format(save_path))


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--input",
        nargs="+",
		help="A list of space separated input images; "
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output. "
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    return parser

if __name__ == "__main__":
	args = get_parser().parse_args()
	logger.add(os.path.join(args.output, 'consolidate.log'))
	os.makedirs(args.output, exist_ok=True)
	main(args)