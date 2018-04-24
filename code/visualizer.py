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
        self.txt_windows = {}
        self.plot_windows = {}  
        
    def make_img_window(self, key):
        self.img_windows[key] = self.vis.images(np.ones((64,3,64,64)))
        
    def make_txt_window(self, key):
        self.txt_windows[key] = self.vis.text(key)
            
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
            
    #
    # push images to visdom
    #
    def show_images(self, key, images):
        self.vis.images(normalize(images), win=self.img_windows[key])
        
    #
    # takes images and their corresponding captions
    # generates html to show in visdom text pane
    #
    def show_text(self, key, captions):
        html = key + "<br><br>"
        
        for i, cap in enumerate(captions):
            html += str(i) + ": " + cap + "<br><br>"
            
        html = "<html>" + html + "</html>"
        self.vis.text(html, win=self.txt_windows[key])
           
    # push new values to visdom plot
    def add_to_plot(self, key, value, count):
        #pdb.set_trace()
        self.vis.line(
            Y = value,
            X = count,
            win = self.plot_windows[key],
            update = 'append'
        )