package loaders_test

import (
	"encoding/binary"
	"io/ioutil"
	"math/rand"
	"os"

	. "github.com/hayden-erickson/neural-network/loaders"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const LABEL_SIZE = 1
const LABEL_HEADER_SIZE = 64

const IMG_HEADER_SIZE = 128
const IMG_ROWS = 28
const IMG_COLS = 28
const IMG_SIZE = IMG_ROWS * IMG_COLS

func randImg() []byte {
	out := make([]byte, IMG_SIZE)

	for i := 0; i < IMG_SIZE/4; i++ {
		binary.BigEndian.PutUint32(out[(i*4):((i+1)*4)], rand.Uint32())
	}

	return out
}

func randImgs(numImgs uint32) []byte {
	out := make([]byte, numImgs*IMG_SIZE)

	for i := 0; i < int(numImgs); i++ {
		copy(out[i*IMG_SIZE:((i+1)*IMG_SIZE)], randImg())
	}

	return out
}

func randLabels(numLabels uint32) []byte {
	out := make([]byte, numLabels)
	for i := 0; i < int(numLabels); i++ {
		out[i] = byte(rand.Intn(10))
	}

	return out
}

var _ = Describe("MnistLoader", func() {
	Describe("#Load", func() {
		var imgfile, labelfile string
		var numItems, imgMagicNum, labelMagicNum uint32
		var numImgs, numLabels uint32
		var labels, images []byte
		var err error
		var l Loader

		BeforeEach(func() {
			imgfile = `test-mnist-data-imgs`
			labelfile = `test-mnist-data-labels`
			numItems = uint32(rand.Intn(100) + 10)
			numImgs = numItems
			numLabels = numItems
		})

		JustBeforeEach(func() {
			imgdata := make([]byte, IMG_HEADER_SIZE+(IMG_SIZE*numImgs))
			labeldata := make([]byte, LABEL_HEADER_SIZE+(LABEL_SIZE*numLabels))

			// img file
			//		magic number header
			imgMagicNum = rand.Uint32()
			binary.BigEndian.PutUint32(imgdata[0:4], imgMagicNum)
			binary.BigEndian.PutUint32(imgdata[4:8], numImgs)
			binary.BigEndian.PutUint32(imgdata[8:12], IMG_ROWS)
			binary.BigEndian.PutUint32(imgdata[12:16], IMG_COLS)
			images = randImgs(numImgs)
			copy(imgdata[16:], images)
			//		create file
			ioutil.WriteFile(imgfile, imgdata, os.ModePerm)

			// label file
			// magic number header
			labelMagicNum = rand.Uint32()
			binary.BigEndian.PutUint32(labeldata[0:4], labelMagicNum)
			binary.BigEndian.PutUint32(labeldata[4:8], numLabels)
			labels = randLabels(numLabels)
			copy(labeldata[8:], labels)

			// create file
			ioutil.WriteFile(labelfile, labeldata, os.ModePerm)
		})

		AfterEach(func() {
			// delete file
			e := os.Remove(imgfile)
			Expect(e).ToNot(HaveOccurred())
			e = os.Remove(labelfile)
			Expect(e).ToNot(HaveOccurred())
		})

		Context("Given filename is incorrect", func() {
			It("returns an error", func() {
				l = NewMNISTLoader(`somefile`, labelfile, imgMagicNum, labelMagicNum)
				_, err = l.Load()
				Expect(err).To(HaveOccurred())

				l = NewMNISTLoader(imgfile, `someotherfile`, imgMagicNum, labelMagicNum)
				_, err = l.Load()
				Expect(err).To(HaveOccurred())
			})
		})

		Context("Given the magic numbers don't match", func() {
			It("returns an error", func() {
				l = NewMNISTLoader(imgfile, labelfile, 94, labelMagicNum)
				_, err = l.Load()
				Expect(err).To(Equal(ErrIncorrectHeader))

				l = NewMNISTLoader(imgfile, labelfile, imgMagicNum, 16)
				_, err = l.Load()
				Expect(err).To(HaveOccurred())
			})
		})

		Context("Given the counts across both files don't match", func() {
			BeforeEach(func() {
				numImgs = numItems + 20
			})

			It("returns an error", func() {
				l = NewMNISTLoader(imgfile, labelfile, imgMagicNum, labelMagicNum)
				_, err = l.Load()
				Expect(err).To(HaveOccurred())
			})
		})

		It("returns the correct examples from file", func() {
			l = NewMNISTLoader(imgfile, labelfile, imgMagicNum, labelMagicNum)
			examples, _ := l.Load()

			Expect(len(examples)).To(Equal(int(numItems)))

			for i, ex := range examples {
				Expect(ex.GetOutput()[labels[i]]).To(BeEquivalentTo(1))
				input := ex.GetInput()

				for j, pixel := range images[(i * IMG_SIZE):((i + 1) * IMG_SIZE)] {
					Expect(input[j]).To(BeEquivalentTo(pixel))
				}
			}
		})
	})
})
