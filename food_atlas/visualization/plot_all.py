from ._performance import plot_performance_box, plot_performance_line
from ._number_of_positive import plot_number_of_positive
from ._calibration import plot_reliability_diagram
from ._curves import plot_curves_all, plot_curves_average


if __name__ == '__main__':
    al_strategies = ['uncertain', 'stratified']
    run_ids = list(range(1, 1 + 100))

    plot_performance_line(
        run_ids,
        active_learning_strategies=al_strategies,
        path_save="outputs/visualization/1a_performance_line.svg",
    )
    plot_performance_box(
        run_ids,
        active_learning_strategies=al_strategies,
        path_save="outputs/visualization/1b_performance_box.svg",
    )
    plot_number_of_positive(
        run_ids,
        active_learning_strategies=al_strategies,
        path_save="outputs/visualization/2_number_of_positive.svg")
    plot_reliability_diagram(
        run_ids=run_ids,
        active_learning_strategies=al_strategies,
        n_bins=5,
        path_save="outputs/visualization/3_reliability_diagram.svg")
    plot_curves_average(
        run_ids=run_ids,
        active_learning_strategies=['uncertain'],
        path_save="outputs/visualization/4a_curves_average.svg")
    plot_curves_all(
        run_ids=run_ids,
        active_learning_strategies=['uncertain'],
        path_save="outputs/visualization/4b_curves_all.svg")
