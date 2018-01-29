package loaders

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io/ioutil"

	"github.com/hayden-erickson/neural-network/nn"
)

var ErrIncorrectHeader = errors.New(`Incorrect Header Value`)
var ErrMismatchedFileSize = errors.New(`File sizes do not match`)

type mNISTExample struct {
	image  []byte
	number byte
}

func (me mNISTExample) GetInput() []float64 {
	out := make([]float64, len(me.image))

	for i, _ := range me.image {
		out[i] = float64(me.image[i])
	}

	return out
}

func (me mNISTExample) GetOutput() []float64 {
	out := make([]float64, 10)
	out[me.number] = 1.0
	return out
}

type mNISTLoader struct {
	imageFile     string
	labelFile     string
	imageMagicNum uint32
	labelMagicNum uint32
}

func NewMNISTLoader(imgs, labels string, imgMagicNum, lblMagicNum uint32) Loader {
	fmt.Println(`new loader`)
	return mNISTLoader{
		imageFile:     imgs,
		labelFile:     labels,
		imageMagicNum: imgMagicNum,
		labelMagicNum: lblMagicNum,
	}
}

func (ml mNISTLoader) Load() ([]nn.Example, error) {
	imgdata, err := ioutil.ReadFile(ml.imageFile)

	if err != nil {
		return []nn.Example{}, err
	}

	labeldata, err := ioutil.ReadFile(ml.labelFile)

	if err != nil {
		return []nn.Example{}, err
	}

	imgMagicNum := binary.BigEndian.Uint32(imgdata[0:4])

	if imgMagicNum != ml.imageMagicNum {
		return []nn.Example{}, ErrIncorrectHeader
	}

	lblMagicNum := binary.BigEndian.Uint32(labeldata[0:4])

	if lblMagicNum != ml.labelMagicNum {
		return []nn.Example{}, ErrIncorrectHeader
	}

	numImgs := binary.BigEndian.Uint32(imgdata[4:8])
	numLabels := binary.BigEndian.Uint32(labeldata[4:8])

	if numImgs != numLabels {
		return []nn.Example{}, ErrMismatchedFileSize
	}

	imgRowSize := binary.BigEndian.Uint32(imgdata[8:12])
	imgColSize := binary.BigEndian.Uint32(imgdata[12:16])
	imgSize := int(imgRowSize * imgColSize)

	// we no longer need the header info
	imgdata = imgdata[16:]
	labeldata = labeldata[8:]
	out := make([]nn.Example, numImgs)

	fmt.Println(`building examples...`)

	for i := 0; i < int(numImgs); i++ {
		out[i] = mNISTExample{
			image:  imgdata[(i * imgSize):((i + 1) * imgSize)],
			number: labeldata[i],
		}
	}

	fmt.Println(`done building examples`)

	return out, nil
}
