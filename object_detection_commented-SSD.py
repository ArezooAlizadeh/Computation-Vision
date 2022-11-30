# Object Detection

# Importing the libraries
import torch  # is best library for neural network and computer vision
from torch.autograd import Variable # autograd is for gradient descent
import cv2 # we are not going to work with the models based on opencv haar cascade. We are just using that because we want to use its rectangle on the detected object
#.. 
from data import BaseTransform, VOC_CLASSES as labelmap # Basetransform convert input image format to neural network format
# VOC_CLASSES is So that's just a very simple dictionary doing the mapping between the text fields of the classes and
# some integers.
from ssd import build_ssd
import imageio # is for some processing on image and applying detect function that we implement on image
# another great library for video processing is PIL

# Defining a function that will do the detections
def detect(frame, net, transform): # We define a detect function that will take as inputs, a frame, a ssd neural network, and a transformation to be applied on the images ( to make the image to have right dimension and format), and that will return the frame with the detector rectangle.
    height, width = frame.shape[:2] # We get the height and the width of the frame.
    frame_t = transform(frame)[0] # We apply the transformation to our frame. It gives two elements but we use [0] since we are interested in first of it
    # so frame_t has right dimension and right color format
    x = torch.from_numpy(frame_t).permute(2, 0, 1) # We convert the frame (numPy array) into a torch tensor
    # a tensor with more advance matrix and advanced array. permute is permutation of color which is small transformation converts sequeence of red blue green
    # into green red blue. X is right torch tensor format with right order of color
    x = Variable(x.unsqueeze(0)) # We add a fake dimension corresponding by unsqueeze to the batch since NN does not accept single input it just accept a batches of images so we add a fake dimesnsion to the image to have it as batch
    # Variable converts torch tensor to torch variable that contains both tensor and gradient. This torch variable would be
    # an element of dynamic graph which allow some fast and efficient computation of gradient during back propagation
    
    y = net(x) # We feed the neural network ssd with the image and we get the output y. Here neural network is pre trained model (with optimized weights)
    # X contains both torch tensor and gradient of the input image. Y 
    
    detections = y.data # We create the detections tensor contained in the output y. data is desired 
    #  y.data is tensor part of torch variable y. Detections is [batch (the first batch we added), numberof classes ( each class belongs to detected object like cat, dog,...), number of occurance of the class ( like two dos in the video), (score, x0,y0,X1,y1)  ]
    # for each occurance of object there is a score with tip left corner and lower right corner. 
    scale = torch.Tensor([width, height, width, height]) # We create a tensor object of dimensions [width, height, width, height]. two of them are upper left corner of image another on eis lower right
    # scale is between 0 and 1
    
    
    for i in range(detections.size(1)): # For every class:   . detections.size(1) is the number of classes
        j = 0 # We initialize the loop variable j that will correspond to the occurrences of the class.
        while detections[0, i, j, 0] >= 0.6: # We take into account all the occurrences j of the class i that have a matching score larger than 0.6.
            pt = (detections[0, i, j, 1:] * scale).numpy() # We get the coordinates of the points at the upper left and the lower right of the detector rectangle.. (255,0,0) is the color of rectangle. 
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) # We draw a rectangle around the detected object. (x0,y0,x1,y1)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA) # We put the label of the class right above the rectangle. like dog , cat, ...
            # i-1: since index in python starts from zero. (int(pt[0]), int(pt[1])) is position of the text
            
            j += 1 # We increment j to get to the next occurrence.
    return frame # We return the original frame with the detector rectangle and the label around the detected object.

# Creating the SSD neural network
net = build_ssd('test') # We create an object that is our neural network ssd.
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage)) # We get the weights of the neural network from another one that is pretrained (ssd300_mAP_77.43_v2.pth).
# torch.load opens weight tensor.  load_state_dict loads the weight to the network

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0)) # We create an object of the BaseTransform class, a class that will do the required transformations so that the image can be the input of the neural network.
# net.size is the target size of the images that feed into neural network
# (104/256.0, 117/256.0, 123/256.0)  scales the color values



# Doing some Object Detection on a video
reader = imageio.get_reader('funny_dog.mp4') # We open the video.
fps = reader.get_meta_data()['fps'] # We get the fps frequence (frames per second).
writer = imageio.get_writer('output.mp4', fps = fps) # We create an output video with this same fps frequence.
# it works for each frame of the video

for i, frame in enumerate(reader): # We iterate on the frames of the output video:
    frame = detect(frame, net.eval(), transform) # We call our detect function (defined above) to detect the object on the frame.
    writer.append_data(frame) # We add the next frame in the output video.
    print(i) # We print the number of the processed frame.
writer.close() # We close the process that handles the creation of the output video.