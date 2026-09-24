/* payload-ledger.c: total what the Phi payload's verbose log says each
 * card and the host did, so "the cards did the work" is a number.
 *
 *   PHI_GGML_VERBOSE=1 scripts/dropper.sh ... 2> run.log
 *   tcc -run tools/payload-ledger.c < run.log
 *
 * Reads "ggml-phi: multiply N: host part H ms, waited W ms more; card C
 * rows R: T ms (pull P, compute K, push U); ..." lines. Prints, per card,
 * the multiplies it took part in and its compute, pull and push time, and
 * for the host its own part and its wait. See payload-ledger.md. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAXC 16

int main(void) {
    static char line[1 << 16];
    long mults = 0, with_cards = 0;
    double host = 0, wait = 0;
    long c_n[MAXC] = {0};
    double c_t[MAXC] = {0}, c_pull[MAXC] = {0}, c_comp[MAXC] = {0}, c_push[MAXC] = {0};
    double c_rows[MAXC] = {0};
    while (fgets(line, sizeof line, stdin)) {
        char *p = strstr(line, "ggml-phi: multiply ");
        if (!p) continue;
        double h, w;
        char *q = strstr(p, "host part ");
        if (!q || sscanf(q, "host part %lf ms, waited %lf ms", &h, &w) != 2) continue;
        mults++;
        host += h;
        wait += w;
        int any = 0;
        for (char *c = strstr(q, "; card "); c; c = strstr(c + 1, "; card ")) {
            int id; long rows; double t, pl, cp, ps;
            if (sscanf(c, "; card %d rows %ld: %lf ms (pull %lf, compute %lf, push %lf)",
                       &id, &rows, &t, &pl, &cp, &ps) == 6 && id >= 0 && id < MAXC) {
                c_n[id]++; c_t[id] += t; c_pull[id] += pl; c_comp[id] += cp; c_push[id] += ps;
                c_rows[id] += rows;
                any = 1;
            }
        }
        with_cards += any;
    }
    printf("{\"multiplies\": %ld, \"with_cards\": %ld, \"host_part_ms\": %.1f, \"host_wait_ms\": %.1f, \"cards\": [",
           mults, with_cards, host, wait);
    int first = 1;
    for (int i = 0; i < MAXC; i++) {
        if (!c_n[i]) continue;
        printf("%s{\"card\": %d, \"multiplies\": %ld, \"rows\": %.0f, \"total_ms\": %.1f, \"compute_ms\": %.1f, \"pull_ms\": %.1f, \"push_ms\": %.1f}",
               first ? "" : ", ", i, c_n[i], c_rows[i], c_t[i], c_comp[i], c_pull[i], c_push[i]);
        first = 0;
    }
    printf("]}\n");
    return 0;
}
