#include <stdio.h>

#define PRN(fmt, ...) fprintf(stdout, fmt "\n", __VA_ARGS__)

void gen_torch_requirements(const char *cuv)
{
    PRN("#!/usr/bin/env -S sh -c 'python3 -m pip install -f %s -r $0'",
        "https://download.pytorch.org/whl/torch_stable.html");
    PRN("torch==%s+%s", "2.1.1", cuv);
    PRN("torchaudio==%s+%s", "2.1.1", cuv);
    PRN("torchvision==%s+%s", "0.16.1", cuv);
}

int main()
{
    gen_torch_requirements("cu118");
    return 0;
}
