package loaders_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"testing"
)

func TestLoaders(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Loaders Suite")
}
