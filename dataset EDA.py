import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols


def import_dataset():
    """Pobiera plik CSV ze zbiorem danych."""
    url = ("https://raw.githubusercontent.com/guipsamora/pandas_exercises/"
           "master/04_Apply/Students_Alcohol_Consumption/student-mat.csv")
    df = pd.read_csv(url)
    print(f" Analizowany zbiór liczy {df.shape[0]} uczniów.")
    return df


def analysis(df):
    """Analizuje dataset - oceny uczniów w zależności od czasu nauki."""
    print("\n[EDA] Krok 1: Statystyki opisowe dla grup")
    eda_result = df.groupby('studytime')['G3'].agg(['count', 'mean',
                                                    'std']).reset_index()
    eda_result.columns = ['Czas nauki (kat. 1-4)', 'Liczba uczniów',
                          'Średnia ocena G3', 'Odchylenie std']

    print(eda_result.to_string(index=False))

    # test
    print("\n[MODEL] Krok 2: Uruchomienie testu ANOVA")

    model = ols('G3 ~ C(studytime)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    p_value = anova_table["PR(>F)"].iloc[0]

    print(f"Uzyskana wartość p-value: {p_value:.5f}")

    if p_value < 0.05:
        print("POTWIERDZAMY HIPOTEZĘ: Różnice istotne statystycznie (p < 0.05).")
        print("Odrzucamy hipotezę zerową. Czas poświęcony na samodzielną naukę "
              "ma realny i zauważalny wpływ na końcowe oceny z matematyki.")
    else:
        print("OBALAMY HIPOTEZĘ: Brak istotności statystycznej (p >= 0.05). "
              "Nie ma podstaw do odrzucenia hipotezy zerowej.")


def plot(df):
    """
    Rysuje wykres pudełkowy zależności ocen końcowych od tygodniowego
    czasu nauki.
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='studytime', y='G3', data=df, hue='studytime',
                palette='Set2', legend=False)
    plt.title('Rozkład ocen końcowych w zależności od tygodniowego czasu nauki')
    plt.xlabel('Kategoria czasu nauki (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)')
    plt.ylabel('Ocena końcowa z matematyki (0-20)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def main():
    import_dataset()
    analysis(import_dataset())
    plot(import_dataset())


if __name__ == '__main__':
    main()