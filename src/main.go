//Using dlib toolkit
//99.4% accuracy on detecting labeled faces
// github.com/daviddao/dlib
// github.com/kagami/go-face
//brew install pkg-config dlib
//sed -i '' 's/^Libs: .*/& -lblas -llapack/' /usr/local/lib/pkgconfig/dlib-1.pc

package main

import (
	"fmt"
	"log"
	"path/filepath"

	"github.com/Kagami/go-face"
)

const dataDir = "../data"//define our data directory

func main() {
	fmt.Println("Facial Recognition System v0.0.1")

	rec, err := face.NewRecognizer(filepath.Join(dataDir, "models"))
	if err != nil {
		fmt.Println("Cannot initialize recognizer")
	}
	defer rec.Close()//don't close our recognizer as it will be closed automatically

	fmt.Println("Recognizer Initialized")
	//count number of faces in the image
	image := filepath.Join(dataDir, "images", "avengers-02.jpeg")//join our data directory with our image

	faces, err := rec.RecognizeFile(image)
	if err != nil {
		log.Fatalf("Can't recognize: %v", err)
	}
	fmt.Println("Total Number of Faces in all Images : ", len(faces))//print number of faces in the image

	//recogize faces in the image based on reference images//machine learning
	var samples []face.Descriptor //array samples of type face.Descriptor
	var avengers []int32//array of indexes of avengers
	for i, f := range faces {//we use these samples to base our future recognitions.
		samples = append(samples, f.Descriptor)
		// Each face is unique on that image so goes to its own category.
		avengers = append(avengers, int32(i))
	}
	// Name the categories, i.e. people on the image.
	labels := []string{
		"Dr Strange",
		"Tony Stark",
		"Bruce Banner",
		"Wong",
	}
	// Pass samples to the recognizer.
	rec.SetSamples(samples, avengers)

	// Now let's try to classify some not yet known image.
	one := filepath.Join(dataDir, "images", "tony-stark.jpg")
	tonyStark, err := rec.RecognizeSingleFile(one)
	if err != nil {
		log.Fatalf("Can't recognize: %v", err)
	}
	if tonyStark == nil {//image is empty
		log.Fatalf("Not a single face on the image or image is empty")
	}
	ID := rec.Classify(tonyStark.Descriptor)
	if ID < 0 {//don't exists in our refernce images data
		log.Fatalf("Can't classify the image")
	}

	fmt.Print("Face ID : ")
	fmt.Print(ID);
	fmt.Print(" Classified as : ")
	fmt.Println(labels[ID]);

	one= filepath.Join(dataDir, "images", "dr-strange.jpg")
	drStrange, err := rec.RecognizeSingleFile(one)
	if err != nil {
		log.Fatalf("Can't recognize: %v", err)
	}
	if drStrange == nil {
		log.Fatalf("Not a single face on the image or image is empty")
	}
	ID = rec.Classify(drStrange.Descriptor)
	if ID < 0 {
		log.Fatalf("Can't classify the image")
	}

	fmt.Print("Face ID : ")
	fmt.Print(ID);
	fmt.Print(" Classified as : ")
	fmt.Println(labels[ID]);

	one= filepath.Join(dataDir, "images", "wong.jpg")
	wong, err := rec.RecognizeSingleFile(one)
	if err != nil {
		log.Fatalf("Can't recognize: %v", err)
	}
	if wong == nil {
		log.Fatalf("Not a single face on the image or image is empty")
	}
	ID = rec.Classify(wong.Descriptor)
	if ID < 0 {
		log.Fatalf("Can't classify the image")
	}
	fmt.Print("Face ID : ")
	fmt.Print(ID);
	fmt.Print(" Classified as : ")
	fmt.Println(labels[ID]);

	one= filepath.Join(dataDir, "images", "avengers-02.jpeg")
	x, err := rec.RecognizeSingleFile(one)
	if err != nil {
		log.Fatalf("Can't recognize: %v", err)
	}
	if x == nil {
		log.Fatalf("Not a single face on the image or image is empty")
	}
	ID = rec.Classify(x.Descriptor)
	if ID < 0 {
		log.Fatalf("Can't classify the image")
	}
	fmt.Print("Face ID : ")
	fmt.Print(ID);
	fmt.Print(" Classified as : ")
	fmt.Println(labels[ID]);
}