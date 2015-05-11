#include <include/Histogram.h>

int main(int argc,
	char** argv)
{
	cv::Mat dummy;
	ghog::lib::Histogram hist(1, dummy);
	return 0;
}
