#include<iostream>
#include "processor.h"

using namespace std;

int main(int argc, char* argv[]){
	auto processor = Processor::getInstance();
	processor->readParameter(argc, argv);
    return 0;
}