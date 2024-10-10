#include<string.h>
#include<assert.h>
#include "circuit/circuit.h"
#include "circuit/operate_unit.h"
#include "tableau/tableau_simluator.h"

class Processor
{
    static Processor* instance;

    // 是否打印量子态
    bool isPrintQState = false;
    // 是否打印时间
    bool isPrintTime = false;
    // 是否打印过程
    bool isPrintProcess = false;

    char* fileName;

    Circuit circuit{};

public:

    static Processor* getInstance() {
        if (!instance) {
            instance = new Processor();
        }
        return instance;
    }

    void PrintQstate() {

    }

    void PrintPProg() {

    }

    void PrintTime() {

    }

    bool readParameter(int argc, char* argv[]) {
        
        if (argc == 1) {
            printf("ERROR: Too Few Parameters.\n");
            return 0;
        } else {
            argv++;
            argc--;
            for (int i = 0; i < argc; ++i) {
                char* now_arg = argv[i];
                if (now_arg[0] == '-') {
                    if(!argParse(now_arg)) {
                        printf("ERROR: Parameter Input Invalid Format.\n");
                        return false;
                    }
                } else {
                    if(!fileParse(now_arg)) {
                        printf("ERROR: File Input Invalid.\n");
                        return false;
                    };
                }
            }
            //TODO 没有输入文件名时的报错
            // printf("ERROR: Please Input File Name.\n");
        }
        return true;
    }

    bool checkFileSuffix(char* suffix) {
        if (suffix[0] == '.'  && suffix[1] == 's' && suffix[2] == 't' && suffix[3] == 'a' && suffix[4] == 'b') return true;
        return false;
    }
    bool fileParse(char* now_arg) {
        assert(strlen(now_arg) > 5);
        char* end = now_arg + strlen(now_arg) - 5;
        uint8_t len = strlen(end);
        if(!checkFileSuffix(end)) {
            return false;
        }
        FILE* file;
        file = fopen(now_arg, "r");
        if(!file) {
            return false;
        }
        readCiruit(file);
        fclose(file);
        return true;
    }

    //
    bool argParse(char* arg) {
        if (strlen(arg) !=2) {
            printf("ERROR: Parameter Input Invalid Format.\n");
            return false;
        }
        char arg_c = arg[1];
        if (arg_c == 'q') {
            isPrintQState = true;
        } else if (arg_c == 't') {
            isPrintTime = true;
        } else if (arg_c == 'p') {
            isPrintProcess == true;
        } else {
            return false;
        }
    }

    // TODO 代码结构更改，该方法应该放到Circuit里面
    bool readCiruit(FILE* fn) {
        uint16_t nums_qbit;
        char c;
        
	    while (!feof(fn)&&(c!='#'))
            fscanf(fn, "%c", &c);
        if (c != '#') {
            printf("ERROR: File Format Error\n");
            return false;
        }   
	    while (!feof(fn))
	    {
            fscanf(fn, "%c", &c);
            GateType gateType;
            size_t qbit;
            if ((c=='\r')||(c=='\n'))
                continue;
            if ((c=='c')||(c=='C')) gateType = GateType::CNOT;
            if ((c=='h')||(c=='H')) gateType = GateType::HADAMARD;
            if ((c=='p')||(c=='P')) gateType = GateType::PHASE;
            if ((c=='x')||(c=='X')) gateType = GateType::X;
            if ((c=='y')||(c=='Y')) gateType = GateType::Y;
            if ((c=='z')||(c=='Z')) gateType = GateType::Z;
            fscanf(fn, "%zu", qbit);
            if (gateType==GateType::CNOT) {
                size_t qbit2;
                fscanf(fn, "%zu", qbit2);
                circuit.addOperation(gateType, qbit, qbit2);
            } else {
                circuit.addOperation(gateType, qbit);
            }
        }
    }   
    
    void doProgram(int argc, char* argv[]) {
        readParameter(argc, argv);
        TableauSimulator tableauSimulator{};
        tableauSimulator.doCircuit(circuit);
    }

};