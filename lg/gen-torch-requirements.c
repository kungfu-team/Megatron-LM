#include <stdio.h>

#define PRN(fmt, ...) fprintf(stdout, fmt "\n", __VA_ARGS__)

#define FPRN(fp, fmt, ...) fprintf(fp, fmt "\n", __VA_ARGS__)

void gen_torch_requirements(FILE *fp, const char *cuv)
{
    FPRN(fp, "#!/usr/bin/env -S sh -c 'python3 -m pip install -f %s -r $0'",
         "https://download.pytorch.org/whl/torch_stable.html");
    FPRN(fp, "torch==%s+%s", "2.1.1", cuv);
    FPRN(fp, "torchaudio==%s+%s", "2.1.1", cuv);
    FPRN(fp, "torchvision==%s+%s", "0.16.1", cuv);
}

int main()
{
    FILE *fp = fopen("requirements.torch.txt", "w");
    gen_torch_requirements(fp, "cu118");
    fclose(fp);
    return 0;
}
