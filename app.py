from flask import Flask, render_template, request
import pickle
import cv2


app = Flask(__name__)

dic = {0:'Pepper_bell_Bacterial_spot', 1:'Pepper_bell_healthy', 2:'Potato_Early_blight', 3:'Potato_healthy', 4:'Tomato_mosaic_virus', 5:'Tomato_healthy' }


model = pickle.load(open('svmodel.pkl','rb'))

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/Disease_Detector", methods=['GET', 'POST'])
def speciesPage():
    return render_template('leaf.html')
    

def predict_label(img_path):
	i = cv2.imread(img_path,0)
	img = cv2.resize(i,(200,200))
	img = img.reshape(1,-1)/255
	p = model.predict(img)
	return dic[p[0]]


@app.route("/submit", methods = ['GET', 'POST'])
def get_hours():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "testing/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)



	return render_template("leaf.html", prediction = p, img_path = img_path)





if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)