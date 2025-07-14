#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include "hmm.h"   


#define MAX_WORDS         1024
#define MAX_WORD_LEN       32
#define MAX_PHONE_LEN       8
#define MAX_PHONE_PER_WORD 16
#define MAX_TOTAL_STATES 10000
#define MAX_T            1024
#define MAX_DIMENSION     39

//*
//조정 가능한 파라미터
double lambda1 = 20.0;   // 언어 모델 weight (bigram 확률 가중치) (양수)
double lambda2 = -3.0;   // word insertion penalty (음수)

//자료구조
typedef struct {
    char word[MAX_WORD_LEN];
    char phones[MAX_PHONE_PER_WORD][MAX_PHONE_LEN];
    int  phone_count;
} DictEntry;

typedef struct {
    char prev[MAX_WORD_LEN];
    char next[MAX_WORD_LEN];
    float prob;
} Bigram;


typedef struct {
    char word[MAX_WORD_LEN];
    int  state_seq[256];
    int  state_len;
} WordHMM;

typedef struct {
    int     start_state;
    int     end_state;
    WordHMM hmm;
} UniversalEntry;


DictEntry      dictionary[MAX_WORDS];
int            dict_size = 0;
Bigram         bigrams[MAX_WORDS * 10];
int            bigram_size = 0;
UniversalEntry universal_words[MAX_WORDS];
int            total_states = 0;

double trans[MAX_TOTAL_STATES][MAX_TOTAL_STATES];

int   init_states[MAX_WORDS];
int   init_size = 0;
float init_probs[MAX_WORDS];

FILE *fp_out;

int state2word[MAX_TOTAL_STATES];
int state2offset[MAX_TOTAL_STATES];

char *recognized_words[100];
int   recognized_len;

//functions
void  load_dictionary(const char *filename);
void  load_bigrams(const char *filename);
void  build_universal_hmm();
void  build_global_transitions();
void  set_initial_states();
double log_sum_exp(double a, double b);
double calc_log_emission(int state, double *vec);
void  viterbi_log(double obs[][MAX_DIMENSION], int T, int *path);
int   read_mfc_file(const char *filename, double obs[][MAX_DIMENSION]);
void  write_mlf_header(FILE *fp);
void  write_recognized_result(FILE *fp, const char *mfc_path, char *words[], int count);
void  extract_words_from_path(int *path, int T);
void  traverse_directory(const char *base_dir, const char *rel_path);
void  run_recognition_model(const char *mfc_base_dir);
int   find_phone_index(const char *pname);

//1. dictionary.txt 읽기
void load_dictionary(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("dictionary.txt open error");
        exit(1);
    }

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        int only_space = 1;
        for (int i = 0; line[i] != '\0'; i++) {
            if (line[i] != ' ' && line[i] != '\t' &&
                line[i] != '\n' && line[i] != '\r') {
                only_space = 0;
                break;
            }

        }
        if (only_space) continue;

        char *first = strtok(line, " \t\n\r");
        if (!first) continue;
        strcpy(dictionary[dict_size].word, first);

        int i_phone = 0;
        char *phone_tok;
        while ((phone_tok = strtok(NULL, " \t\n\r")) != NULL) {
            size_t len = strlen(phone_tok);
            while (len > 0 &&
                   (phone_tok[len - 1] == '\r' || phone_tok[len - 1] == '\n' ||
                    phone_tok[len - 1] == ' '  || phone_tok[len - 1] == '\t')) {
                phone_tok[--len] = '\0';
            }
            if (len == 0) continue;
            strcpy(dictionary[dict_size].phones[i_phone++], phone_tok);
        }
        dictionary[dict_size].phone_count = i_phone;

        if (dictionary[dict_size].phone_count > 0) {
            int last = dictionary[dict_size].phone_count - 1;
            if (strcmp(dictionary[dict_size].phones[last], "sp") == 0) {
                dictionary[dict_size].phone_count--;
            }
        }

        dict_size++;
    }
    fclose(fp);
}

//2. bigram.txt 읽기
void load_bigrams(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("bigram.txt open error");
        exit(1);
    }

    char from[MAX_WORD_LEN], to[MAX_WORD_LEN];
    float prob;
    bigram_size = 0;
    while (fscanf(fp, "%s %s %f", from, to, &prob) == 3) {
        strcpy(bigrams[bigram_size].prev, from);
        strcpy(bigrams[bigram_size].next, to);
        bigrams[bigram_size].prob = prob;
        bigram_size++;
        if (bigram_size >= MAX_WORDS * 10) {
            fprintf(stderr, "Error: bigram size exceeds capacity (%d)\n", MAX_WORDS * 10);
            break;
        }
    }
    fclose(fp);
}

//3. Word, Universal HMM 구성, state→word 매핑핑
int find_phone_index(const char *pname) {
    int numPhones = sizeof(phones) / sizeof(phones[0]);
    for (int i = 0; i < numPhones; i++) {
        if (strcmp(phones[i].name, pname) == 0) return i;
    }
    return -1;
}

typedef struct {
    char word[MAX_WORD_LEN];
    int  state_seq[256];
    int  state_len;
} WordHMM_Helper;

WordHMM_Helper build_word_hmm(DictEntry entry) {
    WordHMM_Helper result;
    strcpy(result.word, entry.word);
    result.state_len = 0;
    for (int i = 0; i < entry.phone_count; i++) {
        char *pname = entry.phones[i];
        int idx = find_phone_index(pname);
        if (idx < 0) continue;
        for (int s = 0; s < N_STATE; s++) {
            result.state_seq[result.state_len++] = idx * N_STATE + s;
        }
    }
    return result;
}





void build_universal_hmm() {
    total_states = 0;
    for (int i = 0; i < dict_size; i++) {
        WordHMM_Helper local = build_word_hmm(dictionary[i]);

        universal_words[i].hmm.state_len = local.state_len;
        strcpy(universal_words[i].hmm.word, local.word);

        universal_words[i].start_state = total_states;
        universal_words[i].end_state   = total_states + local.state_len - 1;

        for (int ofs = 0; ofs < local.state_len; ofs++) {
            int g = total_states + ofs;
            state2word[g]   = i;
            state2offset[g] = ofs;
        }

        for (int k = 0; k < local.state_len; k++) {
            universal_words[i].hmm.state_seq[k] = local.state_seq[k];
        }

        total_states += local.state_len;
    }
}

//4. transitions
void build_global_transitions() {
    for (int i = 0; i < total_states; i++) {
        for (int j = 0; j < total_states; j++) {
            trans[i][j] = 0.0;
        }
    }

    for (int i = 0; i < dict_size; i++) {
        int phone_count = dictionary[i].phone_count;
        for (int k = 0; k < phone_count; k++) {
            const char *phone_name = dictionary[i].phones[k];
            int phone_idx = find_phone_index(phone_name);
            if (phone_idx < 0) continue;

            for (int s = 0; s < N_STATE; s++) {
                for (int s2 = 0; s2 < N_STATE; s2++) {
                    double p = phones[phone_idx].tp[s + 1][s2 + 1];
                    if (p > 0.0) {
                        int g_s  = universal_words[i].start_state + k * N_STATE + s;
                        int g_s2 = universal_words[i].start_state + k * N_STATE + s2;
                        trans[g_s][g_s2] = p;
                    }
                }
            }

            if (k < phone_count - 1) {
                int local_last = N_STATE;
                double p_exit = phones[phone_idx].tp[local_last][local_last + 1];
                const char *next_phone_name = dictionary[i].phones[k + 1];
                int next_phone_idx = find_phone_index(next_phone_name);
                if (next_phone_idx < 0) continue;
                double p_entry = phones[next_phone_idx].tp[0][1];
                double p_combined = p_exit * p_entry;
                int g_last = universal_words[i].start_state + k * N_STATE + (N_STATE - 1);
                int g_next_first = universal_words[i].start_state + (k + 1) * N_STATE + 0;
                trans[g_last][g_next_first] = p_combined;
            }
        }
    }

    for (int i = 0; i < bigram_size; i++) {
        const char *prev_word = bigrams[i].prev;
        const char *next_word = bigrams[i].next;
        float       prob      = bigrams[i].prob;

        int from_indices[MAX_WORDS];
        int from_count = 0;
        for (int d = 0; d < dict_size; d++) {
            if (strcmp(dictionary[d].word, prev_word) == 0) {
                from_indices[from_count++] = d;
            }
        }

        int to_indices[MAX_WORDS];
        int to_count = 0;
        for (int d = 0; d < dict_size; d++) {
            if (strcmp(dictionary[d].word, next_word) == 0) {
                to_indices[to_count++] = d;
            }
        }

        if (from_count == 0 || to_count == 0) continue;

        for (int a = 0; a < from_count; a++) {
            for (int b = 0; b < to_count; b++) {
                int from_idx = from_indices[a];
                int to_idx   = to_indices[b];
                int fs = universal_words[from_idx].end_state;
                int ts = universal_words[to_idx].start_state;
                trans[fs][ts] = prob;
            }
        }
    }
}

//5. <s> → word
void set_initial_states() {
    init_size = 0;
    for (int i = 0; i < bigram_size; i++) {
        if (strcmp(bigrams[i].prev, "<s>") == 0) {
            const char *next_word = bigrams[i].next;
            float prob = bigrams[i].prob;
            for (int d = 0; d < dict_size; d++) {
                if (strcmp(dictionary[d].word, next_word) == 0) {
                    init_states[init_size]  = universal_words[d].start_state;
                    init_probs[init_size++] = prob;
                }
            }
        }
    }
}

//6. Viterbi Algorithm
double log_sum_exp(double a, double b) {
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    if (a > b)      return a + log1p(exp(b - a));
    else            return b + log1p(exp(a - b));
}

double calc_log_emission(int state, double *vec) {
    int i       = state2word[state];
    int ofs     = state2offset[state];
    int id_seq  = universal_words[i].hmm.state_seq[ofs];
    int phone_idx = id_seq / N_STATE;
    int local     = id_seq % N_STATE;

    stateType *s = &phones[phone_idx].state[local];
    double logsum = -INFINITY;

    for (int m = 0; m < N_PDF; m++) {
        double sum = 0.0;
        for (int d = 0; d < MAX_DIMENSION; d++) {
            double diff = vec[d] - s->pdf[m].mean[d];
            sum += (diff * diff) / s->pdf[m].var[d];
        }
        double logprob = log(s->pdf[m].weight) - 0.5 * sum;
        logsum = log_sum_exp(logsum, logprob);
    }
    return logsum;
}

void viterbi_log(double obs[][MAX_DIMENSION], int T, int *path) {
    static double delta[MAX_T][MAX_TOTAL_STATES];
    static int    psi[MAX_T][MAX_TOTAL_STATES];

    for (int i = 0; i < total_states; i++) {
        delta[0][i] = -INFINITY;
    }
    for (int i = 0; i < init_size; i++) {
        int s = init_states[i];
        delta[0][s] = log(init_probs[i]) + calc_log_emission(s, obs[0]);
        psi[0][s]   = -1;
    }

    for (int t = 1; t < T; t++) {
        for (int j = 0; j < total_states; j++) {
            double max_score = -INFINITY;
            int    max_k     = -1;
            for (int k = 0; k < total_states; k++) {
                if (trans[k][j] > 0.0) {
                    int wk = state2word[k];
                    int wj = state2word[j];
                    double transition_logscore;
                    double penalty = 0.0;

                    if (wk == wj) {
                        transition_logscore = log(trans[k][j]);
                    } else {
                        transition_logscore = lambda1 * log(trans[k][j]);
                        penalty = -lambda2;
                    }

                    double score_kj = delta[t - 1][k] + transition_logscore + penalty;
                    if (score_kj > max_score) {
                        max_score = score_kj;
                        max_k     = k;
                    }
                }
            }
            delta[t][j] = max_score + calc_log_emission(j, obs[t]);
            psi[t][j]   = max_k;
        }
    }


    double best_final = -INFINITY;
    int    best_last  = -1;
    for (int i = 0; i < total_states; i++) {
        if (delta[T - 1][i] > best_final) {
            best_final = delta[T - 1][i];
            best_last  = i;
        }
    }
    path[T - 1] = best_last;
    for (int t = T - 2; t >= 0; t--) {
        path[t] = psi[t + 1][path[t + 1]];
    }
}

//7.Read MFCC
int read_mfc_file(const char *filename, double obs[][MAX_DIMENSION]) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("MFCC file open error");
        return 0;
    }
    int T,D;
    fscanf(fp, "%d %d", &T, &D);
    if (D != MAX_DIMENSION) {
        fprintf(stderr, "Dimension mismatch: expected %d, got %d\n", MAX_DIMENSION,D);
        fclose(fp);
        exit(1);
    }
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < D; d++) {
            fscanf(fp, "%lf", &obs[t][d]);
        }
    }
    fclose(fp);
    return T;
}

void write_mlf_header(FILE *fp) {
    fprintf(fp, "#!MLF!#\n");
}



void write_recognized_result(FILE *fp, const char *mfc_path, char *words[], int count) {
    fprintf(fp, "\"%s\"\n", mfc_path);
    for (int i = 0; i < count; i++) {
        fprintf(fp, "%s\n", words[i]);
    }
    fprintf(fp, ".\n");
}

//word sequence로 변환
void extract_words_from_path(int *path, int T) {
    recognized_len = 0;
    int prev_word = -1;
    for (int t = 0; t < T; t++) {
        int st = path[t];
        int w  = state2word[st];

        if (w != prev_word) {
            if (strcmp(dictionary[w].word, "<s>") != 0) {
                recognized_words[recognized_len++] = dictionary[w].word;
            }
            prev_word = w;
        }
    }
    recognized_words[recognized_len] = NULL;
}





void traverse_directory(const char *base_dir, const char *rel_path) {

    char full_dirname[1024];
    if (strlen(rel_path) == 0) {
        snprintf(full_dirname, sizeof(full_dirname), "%s", base_dir);
    } else {
        snprintf(full_dirname, sizeof(full_dirname), "%s/%s", base_dir, rel_path);
    }

    DIR *dir = opendir(full_dirname);
    if (!dir) return;

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, ".."))
            continue;

        char next_rel_raw[1024];
        if (strlen(rel_path) == 0) {
            snprintf(next_rel_raw, sizeof(next_rel_raw), "%s", entry->d_name);
        } else {
            snprintf(next_rel_raw, sizeof(next_rel_raw), "%s/%s", rel_path, entry->d_name);
        }

        char next_full[1024];
        snprintf(next_full, sizeof(next_full), "%s/%s", base_dir, next_rel_raw);

        struct stat st;
        if (stat(next_full, &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            traverse_directory(base_dir, next_rel_raw);
        } else if (S_ISREG(st.st_mode)) {
            const char *filename = entry->d_name;
            size_t len_name = strlen(filename);

            char next_rel_stripped[1024];
            if (len_name > 4 && strcmp(filename + (len_name - 4), ".txt") == 0) {
                char bare_name[256];
                strncpy(bare_name, filename, len_name - 4);
                bare_name[len_name - 4] = '\0';
                if (strlen(rel_path) == 0) {
                    snprintf(next_rel_stripped, sizeof(next_rel_stripped), "%s", bare_name);
                } else {
                    snprintf(next_rel_stripped, sizeof(next_rel_stripped),
                             "%s/%s", rel_path, bare_name);
                }
            } else {
                snprintf(next_rel_stripped, sizeof(next_rel_stripped), "%s", next_rel_raw);
            }

            char mlf_path[1024];
            snprintf(mlf_path, sizeof(mlf_path), "mfc/%s.rec", next_rel_stripped);

            double obs[MAX_T][MAX_DIMENSION];
            int    path[MAX_T];
            int    T = read_mfc_file(next_full, obs);
            if (T <= 0) continue;

            viterbi_log(obs, T, path);

            extract_words_from_path(path, T);
            write_recognized_result(fp_out, mlf_path, recognized_words, recognized_len);
        }
    }
    closedir(dir);
}


// Recognition Model
void run_recognition_model(const char *mfc_base_dir) {
    fp_out = fopen("recognized.txt", "w");
    if (!fp_out) {
        perror("recognized.txt open error");
        exit(1);
    }
    write_mlf_header(fp_out);
    traverse_directory(mfc_base_dir, "");
    fclose(fp_out);
}


//-----------------------------------------------------------------------------------
// 메인 함수

int main() {
    load_dictionary("dictionary.txt");
    load_bigrams("bigram.txt");
    build_universal_hmm();
    build_global_transitions();
    set_initial_states();
    run_recognition_model("/mnt/c/Users/2023user/Desktop/개인자료/KU/2025-1/인공지능/hw02_ASR/mfc"); //코드 작성자 위주로 작성한 경로이니 수정 필요
    return 0;
}
