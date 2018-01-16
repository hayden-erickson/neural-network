package nn_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"testing"
)

func TestNn(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Nn Suite")
}
