import pdb
import visdom
import numpy as np

# normalize imgs to [0,1]
def normalize(nparr):
    mn = nparr.min()
    if mn < 0:
        nparr = nparr + -1 * mn
    mx = nparr.max()
    return nparr / mx

class Visualizer(object):
    
    def __init__(self, servername, portnum, envname):
        self.vis = visdom.Visdom(server=servername, port=portnum, env=envname)
        self.vis.close(env=envname)
        self.img_windows = {}  
        self.plot_windows = {}
        
        # w = self.vis.line(
        #     Y = np.asarray([[1,2,3,4,5]]),
        #     X = np.asarray([[1,1,1,1,1]])
        # )
        # self.vis.line(
        #     Y = np.asarray([[2,3,4,5,6]]),
        #     X = np.asarray([[2,2,2,2,2]]),
        #     win = w,
        #     update = 'append'
        # )
        # self.vis.line(
        #     Y = np.asarray([[2,3,4,5,6]]),
        #     X = np.asarray([[3,3,3,3,3]]),
        #     win = w,
        #     update = 'append'
        # )        
        
        
    def make_img_window(self, key):
        self.img_windows[key] = self.vis.images(np.ones((64,3,64,64)))
            
    def make_plot_window(self, key, num=1, legend=[]):
        
        if num > 1:
            self.plot_windows[key] = self.vis.line(
                X=np.asarray([[0] * num]),
                Y=np.asarray([[0] * num]),
                opts=dict(
                    xlabel=key,
                    legend=legend
                    )
            ) 
        else:
            self.plot_windows[key] = self.vis.line(
                X=np.asarray([0]),
                Y=np.asarray([0]),
                opts=dict(
                    xlabel=key
                    )
            ) 
            
    def show_images(self, key, images):
        self.vis.images(normalize(images), win=self.img_windows[key])
            
    def add_to_plot(self, key, value, count):
        #pdb.set_trace()
        self.vis.line(
            Y = value,
            X = count,
            win = self.plot_windows[key],
            update = 'append'
        )
            
    #
    # shows sentences per image
    # TODO: figure out how to show newlines, 
    # multiple sentences...
    #
    def show_text(self, texts):
        print("todo")
