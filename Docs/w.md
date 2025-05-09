## **零、Storage**

[https://github.com/linnn1103/QT\_Stratagy](https://github.com/linnn1103/QT_Stratagy)

---

## **一、策略核心理念**

* **順勢而行，不猜頂底**  
   價格在單邊趨勢（1H、4H、或 1D）明確展開時，沒碰到更高級別的支撐 / 主力區不嘗試預測反轉，也不硬要找高低點，而是結構續行＋分段進場。

* **高勝率、低風險**  
  透過多時間框架確認趨勢方向，結合結構突破與關鍵區域回踩，配合風險控制，力求單筆交易具有良好 RR。

---

## **二、多時間框架趨勢確認**

1. 在 **4H/1H** 時間框架上，檢視是否出現連續高低點抬升（上漲趨勢）或連續高低點走低（下跌趨勢）。

2. 趨勢明確後，切換至 **15M** 作為主要進出場操作框架；必要時可再下鑽至 **5M/3M** 觀察微結構。

---

## **三、MSB**

* **Market Structure Break（MSB）**：當價格在15M 時間框架出現一波推進段後，突破上一波高點（上漲趨勢）或低點（下跌趨勢），即視為結構續行的「首個訊號」。

* **確認要點**：

  * 突破需伴隨成交量放大或動能指標（如 RSI、MACD）同步上揚 / 走低。

  * 若只是尾端小陰 / 小陽突破，需進一步觀察是否形成乾淨的收盤突破。

---

## **四、關鍵區域回踩觀察**

在完成 MSB 突破後，觀察價格是否回踩以下任一結構區域：

1. **Order Block (OB)**：前期大戶進出貨區域（最後一根趨勢段的始端 K 棒範圍）

2. **Fair Value Gap (FVG)**：單側缺口填補區

3. **Balance Price Range (BPR)**：價格整理高低範圍

4. **重要流動性點**：如前高 / 前低、主要的支撐 / 阻力區、或大級別趨勢線

這些區域往往是機構或大戶防守、補單的重要位置。

---

## **五、回踩反應與進場判斷**

當價格觸及上述區域後，滿足以下任一反應訊號，即可考慮進場：

* **K 線形態**：吞噬、十字星、長下影線 / 長上影線等反轉訊號

* **微結構轉換（CHoCH）**：在更低時間框架（1M/3M/5M）出現高低點結構變化

進場方向與大級別趨勢一致。

---

## **六、風控與分段進場原則**

1. **分段開倉**：每一次 MSB → 回踩 → 反應模式，皆為一次獨立的進場機會，可在同一單邊趨勢中連續加碼。

2. **固定風險**：每筆交易設定明確停損（如區域外 1\~2 個 ATR），風險金額不超過總資金的 5\~10%。

3. **鎖定獲利**：於下一個關鍵結構 / 大級別趨勢線完全平倉，或使用移動式停利（Trailing Stop）鎖定利潤。

---

**七、結構判斷的 Python 實現**

---