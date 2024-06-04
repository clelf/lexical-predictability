import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sns.color_palette("colorblind")


def read_pred_files(files_path):
    # Name the predictability results files start with
    pred_files_name = "pred_scores_trunc64_100samples_level"

    # Create list to append DataFrames to
    dfs = []

    # Select and read pickle files, convert to DataFrame
    for file in os.listdir(files_path):
        if file.startswith(pred_files_name):
            dfs.append(pd.read_csv(os.path.join(files_path, file)))

    # Concatenate data together and return
    return pd.concat(dfs, axis=0, ignore_index=True)



def handle_types(df):
    if "disorder_level" in df.columns:
        df["disorder_level"] = df["disorder_level"].astype(float).apply(lambda x: f"{int(x*100):03d}")
    if "sample_id" in df.columns:
        df["sample_id"] = df["sample_id"].apply(int)
    if "word_pos" in df.columns:
        df["word_pos"] = df["word_pos"].apply(int)
    if "context_length" in df.columns:
        df["context_length"] = df["context_length"].apply(int)
    if "predictability" in df.columns:
        df.rename(columns={'predictability': 'pred_score'}, inplace=True)
        df["pred_score"] = df["pred_score"].apply(float)

    # Sort rows by increasing disorder level
    df.sort_values(by='disorder_level', inplace=True)

    return df

def visualize_predictability(df_preds, save=False):
    # for level in df_preds['disorder_level'].unique():
    #     ax = sns.lineplot(data=df_preds[df_preds['disorder_level'] == level], x='context_length', y='pred_score',
    #                       errorbar='se')
    #     ax.set_title(f'Disorder level: {level}')
    #     plt.show()
        # plt.save(...)

    fig, ax = plt.subplots(layout='constrained', figsize=(7,5))
    sns.lineplot(ax=ax, data=df_preds, hue='disorder_level', x='context_length', y='pred_score', errorbar='se')
    plt.xlabel('Context length')
    plt.ylabel('Single word predictability')
    plt.legend([], [], frameon=False)
    fig.legend(loc='outside center right', title='Disorder level (%)')
    if save:
        plt.savefig("curves_pred_context.png", dpi=600)
        plt.close()
    else:
        plt.show()



def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c


def auc_exp(x, a, b, c):
    return - a / b * np.exp(-b * x) + c * x

def affine_func(x, a, b):
    return a * x + b


def get_fit_params(df, func):
    # Create list to store params to
    params = []

    for level in df.disorder_level.unique():
        df_level = df[df['disorder_level'] == level]

        # Fit curve
        fit_params, _ = curve_fit(func, df_level['context_length'], df_level['pred_score'])

        # Store params
        params.append(
            {'disorder_level': level,
             'params': fit_params}
        )

    return params

def goodness_fit(x, y, fit_params, func):
    # residual sum of squares
    ss_res = np.sum((y - func(x, *fit_params)) ** 2)

    # total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    # r-squared
    r2 = 1 - (ss_res / ss_tot)

    return r2

def plot_pred_context_fit(df, params,  func):
    fig, axs = plt.subplots(2, 5, sharey=True, sharex=True, figsize=(25,10))
    for i, level in enumerate(df.disorder_level.unique()):
        df_level = df[df['disorder_level'] == level]
        params_level = params[i]['params']
        axs.ravel()[i].plot(df_level['context_length'], func(df_level['context_length'], *params_level), '--', label='Fitted curve')
        axs.ravel()[i].plot(df_level['context_length'], df_level['pred_score'], 'ro', label='Measured')
        # axs.ravel()[i].set_title("Disorder level: {}%\n(a={:.2f}, b={:.2f}, c={:.2f})".format(level, *params_level))
        axs.ravel()[i].set_title(f"Disorder level: {level}% \n ({', '.join(['='.join(map(str, i)) for i in zip(['a', 'b', 'c'], [f'{p:0.4f}' for p in params_level])])})")
        # Compute r2
        r2 = goodness_fit(df_level['context_length'], df_level['pred_score'], params_level, func)
        axs.ravel()[i].text(0.1, 0.9, f"r2={r2:.2f}", fontsize=20, transform=axs.ravel()[i].transAxes, va='top')

    fig.supxlabel("Context length")
    fig.supylabel("Predictability")

    axs.ravel()[-1].legend()

    # handles, labels = axs.ravel()[-1].get_legend_handles_labels()
    # leg = fig.legend(handles, labels, loc='outside right center')

    plt.tight_layout()


    if save:
        plt.savefig("curves_fit_pred_context.png", dpi=600, bbox_inches='tight')
        plt.close()
    else:
        plt.show()



def compute_growth(params, func_name, x):
    """
    Given that params is a list containing the parameters a, b, c of the exponential function exp_func,
    the growth is calculated as the initial growth rate, i.e. the derivative of expat x=0, which yields -a*b
    """
    if func_name == 'exp':
        for p in params:
            # 0: a, 1: b, 2: c
            # Integrate
            p['growth'] = auc_exp(x[-1], *p['params']) - auc_exp(x[0], *p['params'])
            # Derive
            # p['rate'] = p['params'][2] / p['params'][1]
    if func_name == 'affine':
        for p in params:
            # 0: a, 1: b
            p['growth'] = p['params'][0]


def plot_fit_level_rate(levels, rates, func):
    # Fit rates to func
    fit_params, _ = curve_fit(func, levels, rates)
    r2 = goodness_fit(levels, rates, fit_params, func)

    # plt.plot(levels, rates)
    plt.plot(levels, func(levels, *fit_params), '--',
                        label='Fitted curve')
    plt.plot(levels, rates, 'ro', label=f'Measured {"AUC" if func == exp_func else "slope factor"}')

    plt.xlabel("Disorder level (%)")
    plt.ylabel("Growth factor")
    plt.xticks(ticks=levels)
    plt.text(0.7, 0.8, f"r2={r2:.2f}", fontsize=20, transform=plt.gca().transAxes, va='top')
    plt.legend()
    plt.title(
        f"{', '.join(['='.join(map(str, i)) for i in zip(['a', 'b', 'c'], [f'{p:0.4f}' for p in fit_params])])}")

    if save:
        plt.savefig("curves_fit_disordelevel.png", dpi=600)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':

    # Define whether to show or save visualizations
    save = False

    # Read data
    df_preds = read_pred_files("preds_all_levels")
    df_preds = handle_types(df_preds)

    # Visualize all levels together
    visualize_predictability(df_preds, save)

    # Analyze relation between growth and disorder level
    # First, group by disorder level, and average all pred. scores per context length value
    df_agg = df_preds.groupby(['disorder_level', 'context_length'])['pred_score'].mean().reset_index()

    func_name = 'exp'
    if func_name == 'exp':
        func = exp_func
    elif func_name == 'affine':
        func = affine_func

    params = get_fit_params(df_agg, func=func)

    plot_pred_context_fit(df_agg, params, func=func)

    # Compare growth slope, relation to disorder level
    compute_growth(params, func_name, df_agg.context_length.sort_values().unique())
    df_growth = pd.DataFrame(params)
    plot_fit_level_rate(df_growth.disorder_level.apply(int), df_growth.growth, func)



