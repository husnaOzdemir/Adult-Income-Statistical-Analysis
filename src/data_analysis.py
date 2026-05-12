import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, shapiro, kstest, jarque_bera, ttest_ind, anderson, normaltest, cramervonmises
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm

# Grafikleri göstermek için fonksiyon
def show_plot():
    plt.show()

# Güvenli örneklem boyutu fonksiyonu (hipotez kodundan alındı)
def get_safe_sample_size(data_series, requested_size):
    return min(requested_size, len(data_series))

# Örneklem boyutunu
sample_size = 30

# Veriyi oku
data = pd.read_csv("../data/adult.data", header=None, na_values=" ?")
data.columns = ["age", "workclass", "fnlwgt", "education", "education-num", 
                "marital-status", "occupation", "relationship", "race", "sex", 
                "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

# Eksik verileri doldur
for col in ["workclass", "occupation", "native-country"]:
    data[col] = data[col].fillna(data[col].mode()[0])
for col in ["age", "fnlwgt", "education-num", "hours-per-week"]:
    data[col] = data[col].fillna(data[col].median())

# Aykırı değerleri temizle
def clean_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

for col in ["education-num", "hours-per-week"]:
    data = clean_outliers(data, col)

# Geliri sayısal yap (0: <=50K, 1: >50K)
data["income_numeric"] = data["income"].map({"<=50K": 0, ">50K": 1})

# Temiz veriyi kaydet
data.to_csv("clean_data.csv", index=False)
data = pd.read_csv("clean_data.csv")

# Sayısal sütunları seç
num_cols = ["age", "fnlwgt", "education-num", "hours-per-week"]

# Veriyi dönüştürme fonksiyonu
def transform_data(column, col_name):
    transforms = {"original": column}
    if column.min() >= 0:
        transforms["log"] = np.log1p(column)  # Log dönüşümü
        transforms["sqrt"] = np.sqrt(column)  # Karekök
    transforms["standard"] = (column - column.mean()) / column.std()  # Standartlaştırma
    return transforms

# Normallik testi fonksiyonu
def check_normality(column, col_name, size, transform="original"):
    print(f"\n=== {col_name} Normallik Testi (n={size}, Dönüşüm: {transform}) ===")
    
    # Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(column, kde=True, bins=20)
    plt.title(f"{col_name} Histogram (n={size}, {transform})")
    plt.xlabel(col_name)
    plt.ylabel("Frekans")
    show_plot()
    
    # Q-Q Grafiği
    plt.figure(figsize=(8, 5))
    sm.qqplot(column, line="s")
    plt.title(f"{col_name} Q-Q Grafiği (n={size}, {transform})")
    show_plot()
    
    # Ortalama, medyan
    mean = column.mean()
    median = np.median(column)
    print(f"Ortalama: {mean:.2f}, Medyan: {median:.2f}")
    if abs(mean - median) < 0.2:
        print("Ortalama ve medyan yakın, dağılım simetrik olabilir.")
    else:
        print("Ortalama ve medyan farklı, dağılım simetrik olmayabilir.")
    
    # Shapiro-Wilk Testi
    stat, p = shapiro(column)
    print(f"Shapiro-Wilk p-değeri: {p:.4f}")
    if p > 0.05:
        print("Normal dağılıma uygun (p > 0.05).")
    else:
        print("Normal dağılıma uygun değil (p ≤ 0.05).")
    
    # Jarque-Bera Testi
    stat, p = jarque_bera(column)
    print(f"Jarque-Bera p-değeri: {p:.4f}")
    if p > 0.05:
        print("Normal dağılıma uygun (p > 0.05).")
    else:
        print("Normal dağılıma uygun değil (p ≤ 0.05).")
    
    # Anderson-Darling Testi
    result = anderson(column, dist='norm')
    stat = result.statistic
    critical_value = result.critical_values[2]  # %5 anlamlılık için
    print(f"Anderson-Darling istatistiği: {stat:.4f}, Kritik değer (α=0.05): {critical_value:.4f}")
    if stat < critical_value:
        print("Normal dağılıma uygun (istatistik < kritik değer).")
    else:
        print("Normal dağılıma uygun değil (istatistik ≥ kritik değer).")
    
    # D'Agostino'nun K² Testi
    stat, p = normaltest(column)
    print(f"D'Agostino K² p-değeri: {p:.4f}")
    if p > 0.05:
        print("Normal dağılıma uygun (p > 0.05).")
    else:
        print("Normal dağılıma uygun değil (p ≤ 0.05).")
    
    # Kolmogorov-Smirnov Testi
    z_scores = (column - mean) / column.std()  
    stat, p = kstest(z_scores, 'norm')
    print(f"Kolmogorov-Smirnov p-değeri: {p:.4f}")
    if p > 0.05:
        print("Normal dağılıma uygun (p > 0.05).")
    else:
        print("Normal dağılıma uygun değil (p ≤ 0.05).")
    
    # Cramér-von Mises Testi
    z_scores = (column - mean) / column.std() 
    result = cramervonmises(z_scores, 'norm')
    stat = result.statistic
    p = result.pvalue
    print(f"Cramér-von Mises p-değeri: {p:.4f}")
    if p > 0.05:
        print("Normal dağılıma uygun (p > 0.05).")
    else:
        print("Normal dağılıma uygun değil (p ≤ 0.05).")
    
    # Sonuç
    normal_tests = [
        p > 0.05 for stat, p in [
            shapiro(column), jarque_bera(column), normaltest(column), 
            kstest(z_scores, 'norm'), (0, result.pvalue)  # Cramér-von Mises
        ]
    ]
    normal_tests.append(stat < critical_value)  # Anderson-Darling
    normal_tests.append(abs(mean - median) < 0.1)  # Ortalama-Medyan
    normal_count = sum(normal_tests)
    print(f"{len(normal_tests)} testten {normal_count} tanesi normal.")
    if normal_count >= 4:  # 7 testten en az 4'ü normal olmalı koşulu
        print(f"{col_name} normal dağılıma yakın.")
    else:
        print(f"{col_name} normal dağılıma uygun değil.")
    return normal_count

# Tam veri seti için normallik testi
print("\n=== Tam Veri Seti Normallik Testleri ===")
for col in num_cols:
    check_normality(data[col], col, len(data))

# Örneklem için normallik testi
print(f"\n=== Örneklem Normallik Testleri (n={sample_size}) ===")
np.random.seed(42)
sample_indices = np.random.choice(data.index, size=sample_size, replace=False)
sample_data = data.loc[sample_indices]

for col in num_cols:
    transforms = transform_data(sample_data[col], col)
    best_transform = "original"
    best_count = -1
    for t_name, t_data in transforms.items():
        count = check_normality(t_data, col, sample_size, t_name)
        if count > best_count:
            best_count = count
            best_transform = t_name
    print(f"\n{col} için en iyi dönüşüm: {best_transform}, Normal test sayısı: {best_count}")

# Yeni Hipotez Testleri (Verilen hipotez kodu uyarlandı)
print("\n=== Hipotez Testleri ===")

# Hipotez testi fonksiyonu 
def run_hypothesis_test(name, description, test_func, data_inputs, alpha, hypotheses, plot_type, **kwargs):
    print(f"\n=== {name} ===")
    print(description)
    print(f"H0: {hypotheses['H0']}")
    print(f"H1: {hypotheses['H1']}")
    print(f"Anlamlılık düzeyi (α): {alpha}")

    try:
        stat, p = test_func(*data_inputs, **kwargs)
        print(f"Test istatistiği: {stat:.4f}, p-değeri: {p:.4f}")
        if p < alpha and not np.isnan(p):
            print(f"Sonuç: p < {alpha}, H0 reddedildi. {hypotheses['H1']}")
        else:
            print(f"Sonuç: p ≥ {alpha}, H0 kabul edildi. {hypotheses['H0']}")
    except Exception as e:
        print(f"Hata: Test sırasında bir sorun oluştu - {str(e)}")
        stat, p = float('nan'), float('nan')

    # Grafik çizimi
    plt.figure(figsize=(8, 5))
    if plot_type == 'barplot':
        labels = ['Maaş >50K', 'Maaş ≤50K']
        if 'data_inputs' in locals() and len(data_inputs) >= 2:
            counts = [data_inputs[0], data_inputs[1] - data_inputs[0]]
            plt.bar(labels, counts, edgecolor='black')
            plt.title(f'{name} (N={data_inputs[1]})')
            plt.ylabel('Kişi Sayısı')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            show_plot()

    elif plot_type == 'boxplot':
        if len(data_inputs) == 2:
            plt.boxplot([data_inputs[0], data_inputs[1]], labels=['Grup 1', 'Grup 2'])
            plt.title(f'{name}')
            plt.ylabel('Haftalık Çalışma Saati')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            show_plot()

# Hipotez testleri listesi
hypothesis_tests = [
    {
        'name': 'Eğitim Seviyesine Göre Maaş Oranı',
        'description': (
            "Bir şirket, eğitim seviyesi 12’den yüksek olan çalışanların %25’ten fazlasının maaşının 50,000 TL üzeri (>50K) olduğunu iddia etmektedir. "
            "100 eğitim seviyesi 12’den yüksek çalışandan 30’unun maaşının 50K üzeri olduğu belirlenmiştir. "
            "Bu iddianın %5 anlamlılık düzeyinde doğru olup olmadığını test edin."
        ),
        'test_func': proportions_ztest,
        'data_inputs': [30, 100, 0.25],
        'kwargs': {'alternative': 'larger'},
        'alpha': 0.05,
        'hypotheses': {
            'H0': 'Eğitim seviyesi 12’den yüksek çalışanların %25’i yüksek maaş (>50K) almaktadır (p = 0.25).',
            'H1': 'Eğitim seviyesi 12’den yüksek çalışanların %25’ten fazlası yüksek maaş (>50K) almaktadır (p > 0.25).'
        },
        'plot_type': 'barplot'
    },
    {
        'name': 'Eğitim Seviyesine Göre Çalışma Saati',
        'description': (
            "Bir şirket, eğitim seviyesi 12’den yüksek olan çalışanların haftalık ortalama çalışma saatlerinin, eğitim seviyesi 12 veya daha düşük olan çalışanlardan "
            "daha fazla olduğunu iddia etmektedir. Veri setinden her gruptan 80’er çalışanın haftalık çalışma saatleri rastgele seçilmiştir. "
            "Bu iddianın %5 anlamlılık düzeyinde doğru olup olmadığını test edin."
        ),
        'test_func': ttest_ind,
        'data_inputs': [
            data[data['education-num'] > 12]['hours-per-week'].sample(get_safe_sample_size(data[data['education-num'] > 12]['hours-per-week'], 80), random_state=42),
            data[data['education-num'] <= 12]['hours-per-week'].sample(get_safe_sample_size(data[data['education-num'] <= 12]['hours-per-week'], 80), random_state=42)
        ],
        'kwargs': {'alternative': 'greater'},
        'alpha': 0.05,
        'hypotheses': {
            'H0': 'Eğitim seviyesi 12’den yüksek ve düşük olan çalışanların haftalık ortalama çalışma saatleri eşittir (μ1 = μ2).',
            'H1': 'Eğitim seviyesi 12’den yüksek olan çalışanların haftalık ortalama çalışma saati, düşük olanlardan fazladır (μ1 > μ2).'
        },
        'plot_type': 'boxplot'
    },
    {
        'name': 'Yaş Grubuna Göre Çalışma Saati',
        'description': (
            "Bir kuruluş, 35 yaşından büyük çalışanların haftalık ortalama çalışma saatlerinin, 35 yaş ve altı çalışanlardan daha fazla olduğunu iddia etmektedir. "
            "Veri setinden her gruptan 70’er çalışanın haftalık çalışma saatleri rastgele seçilmiştir. "
            "Bu iddianın %5 anlamlılık düzeyinde doğru olup olmadığını test edin."
        ),
        'test_func': ttest_ind,
        'data_inputs': [
            data[data['age'] > 35]['hours-per-week'].sample(get_safe_sample_size(data[data['age'] > 35]['hours-per-week'], 70), random_state=42),
            data[data['age'] <= 35]['hours-per-week'].sample(get_safe_sample_size(data[data['age'] <= 35]['hours-per-week'], 70), random_state=42)
        ],
        'kwargs': {'alternative': 'greater'},
        'alpha': 0.05,
        'hypotheses': {
            'H0': '35 yaşından büyük ve küçük çalışanların haftalık ortalama çalışma saatleri eşittir (μ1 = μ2).',
            'H1': '35 yaşından büyük çalışanların haftalık ortalama çalışma saati, küçük olanlardan fazladır (μ1 > μ2).'
        },
        'plot_type': 'boxplot'
    },
    {
        'name': 'Haftalık Çalışma Saatine Göre Maaş Oranı',
        'description': (
            "Bir şirket, haftada 40 saatten fazla çalışanların %20’sinden fazlasının maaşının 50K üzeri (>50K) olduğunu iddia etmektedir. "
            "120 haftada 40 saatten fazla çalışan çalışandan 28’inin maaşının 50K üzeri olduğu belirlenmiştir. "
            "Bu iddianın %5 anlamlılık düzeyinde doğru olup olmadığını test edin."
        ),
        'test_func': proportions_ztest,
        'data_inputs': [28, 120, 0.20],
        'kwargs': {'alternative': 'larger'},
        'alpha': 0.05,
        'hypotheses': {
            'H0': 'Haftada 40 saatten fazla çalışanların %20’si yüksek maaş (>50K) almaktadır (p = 0.20).',
            'H1': 'Haftada 40 saatten fazla çalışanların %20’sinden fazlası yüksek maaş (>50K) almaktadır (p > 0.20).'
        },
        'plot_type': 'barplot'
    },
    {
        'name': 'Yaş ve Maaş Oranı',
        'description': (
            "Bir kuruluş, 40 yaşından küçük çalışanların %15’inden fazlasının maaşının 50K üzeri (>50K) olduğunu iddia etmektedir. "
            "80 adet 40 yaşından küçük çalışandan 15’inin maaşının 50K üzeri olduğu belirlenmiştir. "
            "Bu iddianın %5 anlamlılık düzeyinde doğru olup olmadığını test edin."
        ),
        'test_func': proportions_ztest,
        'data_inputs': [15, 80, 0.15],
        'kwargs': {'alternative': 'larger'},
        'alpha': 0.05,
        'hypotheses': {
            'H0': '40 yaşından küçük çalışanların %15’i yüksek maaş (>50K) almaktadır (p = 0.15).',
            'H1': '40 yaşından küçük çalışanların %15’inden fazlası yüksek maaş (>50K) almaktadır (p > 0.15).'
        },
        'plot_type': 'barplot'
    }
]

# Hipotez testlerini çalıştır
for i, test in enumerate(hypothesis_tests, 1):
    print(f"\n{i}. Test")
    run_hypothesis_test(
        name=test['name'],
        description=test['description'],
        test_func=test['test_func'],
        data_inputs=test['data_inputs'],
        alpha=test['alpha'],
        hypotheses=test['hypotheses'],
        plot_type=test['plot_type'],
        **test['kwargs']
    )
