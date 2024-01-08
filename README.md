# 1. **タイトルとバージョン**
    - タイトル: マッチング選出
    - バージョン: 1.0.0

# 2. **導入**
    - ソフトウェアの概要: 
  本プログラムは、マッチング理論を活用して、募集する役職と、役職を希望する会員を適切にマッチさせるためのツールです。小学校のPTA役員選出を目的に作成しましたが、その他にも様々な場面で使用できます。具体的な流れとしては、役員選出のアンケートをGoogleフォームなどで収集し、それと併せて募集する役職と人数を設定したファイルを入力情報として利用します。マッチングの条件として、希望する役職や、その他の条件（例: 過去の役員経験、WordやExcelのスキルレベル）をスコア化し、高得点者を優先的に選出します。同点の場合は、ランダム性のある選出をするために、ID（例: アンケートの回答日、学年、クラス、出席番号）などからハッシュ値を求め、そのハッシュ値の大きい人を選出します。この方式により、同じ入力情報でプログラムを実行すると、常に同一の結果が得られる特性を持っています。さらに、各役職ごとの補欠選出も可能で、補欠の人数を設定することができます。

    - ソフトウェアの主な機能と利点: 
  従来の小学校のPTA役員選出は、くじ引きや推薦によって行われることが一般的でしたが、それは準備が大変で多くの人々の時間や体力を消費していました。現代のデジタル化が進む中、多くの学校はGoogleフォームを用いてアンケートを実施することが増えてきましたが、それに続く選出作業は手動での作業が中心であった。このソフトウェアを採用することにより、明確な選出ロジックを元に、Googleフォームからのアンケート結果を瞬時に処理して選出作業を行うことができるようになります。

# 3. **システム要件**
    - OS: Windows 11

# 4. **インストールガイド**
    - インストール手順: インストールは不要です。
    - SmartScreen警告についてのご案内
        - 当ソフトウェアをダウンロードまたは実行する際、WindowsのSmartScreenによる警告が表示される場合があります。
          これは、当ソフトウェアがまだ多くのユーザーに認知されていないために発生するもので、当ソフトウェアに悪意があるわけではございません。
                - 開発言語: Python
                - パッケージ化ツール: pyinstallerを使用し、.exe形式に変換しています。
        - 警告を無視する手順:
          警告が表示された場合、[詳細] をクリックします。
          [実行] または [続行] のオプションが表示されるので、それをクリックしてインストールを続行してください。
    - 条件設定: `config.ini` に以下の情報を記載してください。

        **[FILES]**
        - input_file: アンケート結果のCSVファイル名
        - output_file: マッチング結果のCSVファイル名
        ```
        例: input_file=sample.csv
        例: output_file=matching_results.csv
        ```

        **[ENCODING]**
        - input_encoding: 入力ファイルのエンコード
        - output_encoding: 出力ファイルのエンコード
        ```
        例:
        input_encoding=utf-8
        output_encoding=utf_8_sig

        **[RECRUIT]**
    - 役職と募集人数
        ```
        例:
        会長=1
        副会長=1
        書記=3
        会計=2
        ```

        **[SUPPLEMENTARY_RECRUIT]**
    - 補欠募集の役職と募集人数
        ```
        例:
        会長=2
        副会長=2
        書記=6
        会計=4
        ```

        **[IDS]**
    - ID用項目（CSVファイルの列番号を指定）
        ```
        例:
        id1=1
        id2=2
        ```

        **[PARAMETERS]**
    - スコア算出項目（CSVファイルの列番号を指定）
        ```
        例:
        param1=3
        param2=4
        ```

        **[CHOICES]**
    - 希望役職項目（CSVファイルの列番号を指定）
        ```
        例:
        choice1=5
        choice2=6
        choice3=7
        ```

        **[POSITION_WEIGHTS]**
    - 各役職のスコア算出用の重みづけ（`param1` は `weight1`、`param2` は `weight2` と対応）
        ```
        注意: アンケートが「1.やりたい、2.どちらでもいい、3.やりたくない」という形式の場合、
        「1.やりたい」を高スコアにするには、重みづけを負の値に設定する必要があります。
        例: weight1=-5
        weight2=1
        ```

        同様のセクションを `副会長WEIGHTS`、`書記WEIGHTS`、`会計WEIGHTS` としても設定してください。


# 5. **基本的な操作方法**
- マッチングの実行: `matching.exe` をダブルクリックして実行してください。
- 結果の確認: マッチングの結果は `matching_results.csv` に出力されます。当選者の役職は `Elected` 列に、補欠者の役職は `Alternate` 列に表示されます。

# 6. **詳細機能**

- データの正規化:
  - アンケート回答データの正規化について: 各データ値からその列の中央値を減算し、その結果をその列の最大値で除算します。以下に、この正規化手法の特性と利点を説明します。
    - 中心の移動: メディアンはデータの中央値を示し、外れ値の影響を受けにくい特性があります。特に、外れ値を含むデータや非対称な分布の場合、メディアンを参照点として利用するのが適切です。
    - スケーリング: 最大値で除算することで、データは-1から1の範囲に収められます。これにより、異なる尺度の特徴を同一のスケールに揃えることができます。
    - 外れ値の影響: 最大値によるスケーリングでは、外れ値の影響を考慮する必要があります。外れ値が非常に大きい場合、スケーリングの上限がその外れ値によって決まります。

  例として、あるデータセット D における各データポイント x_i の正規化後の値 x'_i は以下のように計算されます。

  x'_i = (x_i - median(D)) / max(D)

  ここで:
  - x'_i は正規化後のデータポイントの値です。
  - x_i は正規化前のデータポイントの値です。
  - median(D) はデータセット D の中央値（メディアン）です。
  - max(D) はデータセット D の最大値です。

- データの重みづけ:
  - 例えば、データセットの変数として次のように定義します。
    - モチベーション: A
    - 過去の役員経験回数: B
    - PTA活動に割ける時間: C
    - Wordのスキル: D
    - Excelのスキル: E
    - Googleドライブのスキル: F
  - これらの変数を前述した方法で正規化した後、以下の数式を使用してスコアを算出します。
        スコア = w_A * A' + w_B * B' + w_C * C' + w_D * D' + w_E * E' + w_F * F'

         ここで:
          - A', B', C', ... は正規化後のデータポイントの値です。
         - w_A, w_B, w_C, ... は各正規化後のデータポイントの値に対する重みづけです。

  - この方法で、スコアのデータセットが構築されます。役職ごとに重みづけを変更することで、より適切なマッチングが期待できます。

- ハッシュ値による選出:
  - スコアが同一となる場合のマッチングを公平かつ透明に行うため、アンケート回答結果をsha256でハッシュ値に変換し、そのハッシュ値を基に選出を行います。
  - この方法には、ランダム選出とは異なり、もう1度選出作業を行っても常に同じ結果となるという特徴があります。
  - さらに、ハッシュ値はアンケート回答結果に対して一意に決定されます。選出者の意図や主観が一切関与しないため、客観的なアプローチにより、公正な選出が保証されます。

# 7. **設定とカスタマイズ**

    - 設定メニュー: 本ソフトウェアのすべての設定は`config.ini`ファイルを通じて行います。このファイルで各種パラメーターや設定項目を編集することで、ソフトウェアの動作をカスタマイズできます。

    - ソフトウェアのカスタマイズ方法: 詳細なカスタマイズ方法や設定項目の解説については、**4. インストールガイド**を参照してください。こちらに具体的な手順や設定項目の詳細が記載されています。


# 8. **連絡先**

    - サポート連絡先: ソフトウェアに関するサポートが必要な場合、GitHubの「Issues」セクションで新しいIssueを作成してください。

    - フィードバック: ソフトウェアに関するフィードバックや改善の提案がある場合も、GitHubの「Issues」セクションで新しいIssueを作成してください。あなたのご意見は、ソフトウェアの品質向上に役立てさせていただきます。



# 9. **ライセンスと著作権情報**

    - 使用許諾: 本ソフトウェアはオープンソースです。商用利用を含む任意の用途での使用、改変、再配布が許可されます。

    - 免責事項: 本ソフトウェアを使用した結果に関してのいかなる保証も提供されません。また、本ソフトウェアによって生じたいかなる損害に対しても開発者は責任を負いません。
　　　　　　　　 使用者は自己の責任において本ソフトウェアを利用するものとします。

    - 著作権情報: 本ソフトウェアの著作権は開発者に帰属します。しかし、ソースコードやバイナリ、関連ドキュメントの改変、再配布は許可されています。

    - その他: 本ソフトウェアを使用する際の連絡や報告は特に必要ございません。


# 10. **用語集**

    - マッチング理論 (Matching Theory): 経済学の一分野で、市場の参加者たちがペアを組むプロセスやその結果を研究する理論。

    - ゲール＝シャープレー・アルゴリズム (Gale-Shapley Algorithm): マッチング理論の中で、安定的なマッチングを求めるためのアルゴリズム。
　　　参加者がそれぞれリストを元に希望する相手にプロポーズを行い、最終的に受け入れる相手を選ぶ方法でマッチングを行う。

    - 正規化 (Normalization): データを特定の範囲や標準に変換する処理。この文書では、データを-1から1の範囲に収めるための処理を指す。

    - メディアン (Median): あるデータセットの中央値。データが小さい順に並べたときに中央にくる値。外れ値の影響を受けにくいので、データの中央の参照点としてしばしば用いられる。

    - スケーリング (Scaling): データの範囲を変更すること。通常は、異なる特徴量を同一の尺度や範囲に揃えるために行われる。

    - 外れ値 (Outlier): 他の値と比べて極端に大きいまたは小さい値。データの解析や処理に影響を与える可能性がある。

    - 重みづけ (Weighting): ある要素や値に対する重要度や影響度を数値で示すこと。この文書では、正規化後のデータポイントの値に対する重みづけを指す。

    - CSV (Comma-Separated Values): データをカンマで区切って表現するテキスト形式。表形式のデータを保存や交換する際に広く使われている。

    - config.ini: ソフトウェアの設定やパラメータを保存するためのファイル。INI形式で書かれる。



