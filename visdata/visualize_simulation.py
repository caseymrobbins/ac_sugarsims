import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# -------------------------------
# Load parquet
# -------------------------------

def load_data(path):
    df = pd.read_parquet(path)
    print("\nLoaded dataset")
    print("-----------------------")
    print(df.shape)
    print("\nColumns:")
    for c in df.columns:
        print(c)
    return df


# -------------------------------
# Basic time series plot
# -------------------------------

def plot_metric(df, metric):

    if metric not in df.columns:
        print("Metric not found")
        return

    plt.figure(figsize=(10,5))
    plt.plot(df["step"], df[metric])
    plt.title(metric)
    plt.xlabel("Step")
    plt.ylabel(metric)
    plt.grid(True)
    plt.show()


# -------------------------------
# Collapse dashboard
# -------------------------------

def collapse_dashboard(df):

    fig, ax = plt.subplots(2,2, figsize=(12,8))

    ax[0,0].plot(df.step, df.agency_floor)
    ax[0,0].set_title("Agency Floor")

    ax[0,1].plot(df.step, df.unemployment_rate)
    ax[0,1].set_title("Unemployment")

    ax[1,0].plot(df.step, df.hhi)
    ax[1,0].set_title("Firm Concentration (HHI)")

    ax[1,1].plot(df.step, df.horizon_index)
    ax[1,1].set_title("Horizon Index")

    plt.tight_layout()
    plt.show()


# -------------------------------
# Correlation heatmap
# -------------------------------

def correlation_map(df):

    numeric = df.select_dtypes(include="number")

    corr = numeric.corr()

    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Metric Correlations")
    plt.show()


# -------------------------------
# Multiple metrics
# -------------------------------

def multi_plot(df, metrics):

    plt.figure(figsize=(10,6))

    for m in metrics:
        if m in df.columns:
            plt.plot(df.step, df[m], label=m)

    plt.legend()
    plt.xlabel("Step")
    plt.title("Multiple Metrics")
    plt.show()


# -------------------------------
# Menu interface
# -------------------------------

def menu(df):

    while True:

        print("\nOptions")
        print("1  Plot single metric")
        print("2  Collapse dashboard")
        print("3  Correlation heatmap")
        print("4  Multi metric plot")
        print("5  List columns")
        print("0  Exit")

        choice = input("Choice: ")

        if choice == "1":

            metric = input("Metric name: ")
            plot_metric(df, metric)

        elif choice == "2":

            collapse_dashboard(df)

        elif choice == "3":

            correlation_map(df)

        elif choice == "4":

            m = input("Metrics (comma separated): ")
            metrics = [x.strip() for x in m.split(",")]
            multi_plot(df, metrics)

        elif choice == "5":

            for c in df.columns:
                print(c)

        elif choice == "0":
            break


# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("python visualize_simulation.py data.parquet")
        sys.exit()

    file = sys.argv[1]

    df = load_data(file)

    menu(df)