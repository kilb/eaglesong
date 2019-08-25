#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int bit_matrix[] = {1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
                    0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
                    0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1,
                    0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0,
                    1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1,
                    1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0,
                    1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1,
                    0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1,
                    0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1,
                    0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
                    0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
                    1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0,
                    0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                    0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
                    1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1};

int vec1[16];
int vec2[16];
int path[16];
int path2[16][16];

int find(int i, int j, int tmp, int n) {
    if(tmp == vec2[i]) {
        for(int k=0; k<16; ++k) {
            printf("%d[%d] ", k, path[k]);
            path2[i][k] = path[k];
        }
        printf(", %d\n", n);
        return 1;
    }
    if(j == 16) {
        return 0;
    }

    path[j] = 0;
    find(i, j+1, tmp, n);
    path[j] = 1;
    find(i, j+1, tmp ^ vec1[j], n+1);
    path[j] = 0;
}

int main() {
    int tmp = 1;
    for(int i=0; i<16; ++i) {
        vec1[i] = tmp;
        tmp <<= 1;
    }

    for(int i=0; i<16; ++i) {
        int tmp=0;
        for(int j=15; j>=0; --j) {
            tmp = (tmp << 1) + bit_matrix[j*16+i];
        }
        vec2[i] = tmp;
        printf("%d\n", vec2[i]);
    }

    for(int i=0; i<16; ++i) {
        memset(path, 0, sizeof(int)*16);
        find(i, 0, 0, 0);
        // printf("------------\n");
        memset(path, 0, sizeof(int)*16);
        find(i+1, 0, 0, 0);

        for(int j=0; j<16; ++j) {
            if(path2[i][j] && path2[i+1][j]) {
                printf("%d, ", j);
            }
        }
        printf("\n");
        for(int j=0; j<16; ++j) {
            if(path2[i][j] && !path2[i+1][j]) {
                printf("%d, ", j);
            }
        }
        printf("\n");
        for(int j=0; j<16; ++j) {
            if(!path2[i][j] && path2[i+1][j]) {
                printf("%d, ", j);
            }
        }
        printf("\n");
        printf("------------\n");
        vec1[i] = vec2[i];
        ++i;
        vec1[i] = vec2[i];
    }

    for(int i=0; i<16; ++i) {
        printf("s%d = state[%d]; ", i,i);
        ++i;
        printf("s%d = state[%d]; ", i,i);
        ++i;
        printf("s%d = state[%d]; ", i,i);
        ++i;
        printf("s%d = state[%d];\n", i,i);
    }

    for(int i=0; i<16; ++i) {
        printf("state[%d] = s%d; ", i,i);
        ++i;
        printf("state[%d] = s%d; ", i,i);
        ++i;
        printf("state[%d] = s%d; ", i,i);
        ++i;
        printf("state[%d] = s%d;\n", i,i);
    }



}