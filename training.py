class TrainFacesAlg:
    def __init__(self, tenant_code, roll_number, user_id=None):
        self.tenant_code = str(tenant_code)
        self.roll_number = str(roll_number)
        self.cascPath = settings.BASE_DIR + "/media/haarcascades/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(self.cascPath)
        
        self.user_face_dir = settings.BASE_DIR + "/media/uploaded_photos/" + self.tenant_code + "/" + self.roll_number
        self.user_trained_dir = settings.BASE_DIR + "/media/trained_photos/" + self.tenant_code + "/" + self.roll_number
        self.RESIZE_FACTOR = 4
        self.user_id = user_id
        
        

    def train_img(self):
        imgs = []
        tags = []
        Path(settings.BASE_DIR + "/media/uploaded_photos/" + self.tenant_code).mkdir(parents=True, exist_ok=True)
        try:
            images_available = [(os.remove(self.user_trained_dir + "/" + f)) for f in os.listdir(self.user_trained_dir) if os.path.isfile(os.path.join(self.user_trained_dir, f))]
        except Exception:
            pass
        Path(self.user_trained_dir).mkdir(parents=True, exist_ok=True)
        resized_width, resized_height = (112, 92)
        index = 1
        for img_file_name in os.listdir(self.user_face_dir):
            if ".txt" in img_file_name:
                continue
            image_path =  self.user_face_dir + "/" + img_file_name
            image_data = cv2.imread(image_path)
            grayscale_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            gray_resized_img = cv2.resize(grayscale_image, (round(grayscale_image.shape[1] / self.RESIZE_FACTOR), round(grayscale_image.shape[0] / self.RESIZE_FACTOR)))
            faces = self.face_cascade.detectMultiScale(
                gray_resized_img,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
                )
            if len(faces) == 1 :
                areas = []
                for(x,y,w,h) in faces:
                    areas.append(w * h)
                max_area, idx = max([(val, idx) for idx, val in enumerate(areas)])
                face_sel = faces[idx]
                x = face_sel[0] * self.RESIZE_FACTOR
                y = face_sel[1] * self.RESIZE_FACTOR
                w = face_sel[2] * self.RESIZE_FACTOR
                h = face_sel[3] * self.RESIZE_FACTOR
                face = grayscale_image[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (resized_width, resized_height))
                saved_img = self.user_trained_dir + '/' + img_file_name
                cv2.imwrite(saved_img, face_resized)
                imgs.append(cv2.imread(saved_img, 0))
                tag = str(self.user_id) + str(index)
                tags.append(int(tag))
                index = index + 1
        (imgs_list, tags_list) = [np.array(item) for item in [imgs, tags]]
        if len(tags) > 0 and len(imgs) > 0:
            f = open(self.user_trained_dir + '/user.txt', 'w')
            f.write('{}'.format(self.user_id))
            f.close()
            f = open(self.user_face_dir + '/user.txt', 'w')
            f.write('{}'.format(self.user_id))
            f.close()
            if len(tags) == 1 and len(imgs) == 1:
                tags.append(tags[0]+1)
                imgs.append(imgs[0])
                (imgs_list, tags_list) = [np.array(item) for item in [imgs, tags]]
            
            self.lbphface_model = cv2.face.LBPHFaceRecognizer_create()
            self.lbphface_model.train(imgs_list, tags_list)
            self.lbphface_model.save(self.user_trained_dir + '/lbphface_trained_data.xml')

            self.fisherface_model = cv2.face.FisherFaceRecognizer_create()
            self.fisherface_model.train(imgs_list, tags_list)
            self.fisherface_model.save(self.user_trained_dir + '/fisherface_trained_data.xml')
            
            self.eigenface_model = cv2.face.EigenFaceRecognizer_create()
            self.eigenface_model.train(imgs_list, tags_list)
            self.eigenface_model.save(self.user_trained_dir + '/eigenface_trained_data.xml')
            return True
        else:
            return False
