from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import torchvision.transforms as transforms


def run_alignment(image_path):
  import dlib
  from scripts.align_all_parallel import align_face, align_face_dfl
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  print(image_path)
  aligned_image = align_face_dfl(filepath=image_path, predictor=predictor) 
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image 
  

class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]

		# #CL: For FFHQ images you only need to read as they are 
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

		# CL: For wf images extracted from DFL, you need to rerun alignment
		# from_im = run_alignment(from_path)

		if self.transform:
			out_im = self.transform(from_im)

		# CL: self.transform resizes from_im to 256 (for running encoding)
		# So we keep a copy of high resolution input for comparing with the output
		my_transform = transforms.Compose([transforms.Resize((1024, 1024)),
										   transforms.ToTensor(),
										   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
		from_im = my_transform(from_im)

		return out_im, from_im
