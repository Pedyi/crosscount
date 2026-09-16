"""Run every experiment in the order the paper reports them."""
import run_crossc4, run_diagnosis, run_diagnostic, run_learned
import run_main, run_robustness, run_synthetic, run_validation

if __name__ == '__main__':
    for step in [run_synthetic, run_diagnostic, run_main, run_validation,
                 run_learned, run_diagnosis, run_crossc4, run_robustness]:
        print(f'\n{"=" * 70}\n{step.__name__}\n{"=" * 70}')
        step.main()
