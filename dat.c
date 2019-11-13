
#include <stdio.h> 
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
	FILE *file;
			int a = atoi(argv[1]);
			int b = atoi(argv[2]);
    	file = fopen(argv[3],"rb");

	if(file == NULL)
	{
		fprintf(stdout, ".dat failed to open.  exiting...\n");
		exit(1);
	}
      	
	int count=0;
	
	float v;
	float vet[a*b];
	while(!feof(file)){
		fread((void*)(&v), sizeof(v), 1, file);
		vet[count]=v;
		count++;
	}
	
	for(int i=0; i<a*b; i++){
		if (i%b == 0) printf("\n");
	 	printf("%f\n", vet[i]);
	}

	fclose(file);
}
