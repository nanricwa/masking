# テスト用の「操作録画っぽい」動画と画像を作る（架空の個人情報）
import cv2, numpy as np, sys
from PIL import Image, ImageDraw, ImageFont
F = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
font = ImageFont.truetype(F, 26); small = ImageFont.truetype(F, 20)
def screen(scroll=0, typed=""):
    im = Image.new("RGB", (1280, 720), (245, 246, 248)); d = ImageDraw.Draw(im)
    d.rectangle([0,0,1280,56], fill=(40,70,140)); d.text((24,12),"顧客管理システム ─ 顧客詳細", font=font, fill="white")
    rows = [("氏名","山田 太郎"),("フリガナ","ヤマダ タロウ"),("電話番号","090-1234-5678"),
            ("メール","taro.yamada@example.co.jp"),("住所","東京都渋谷区神南1-2-3 テストビル5F"),
            ("郵便番号","〒150-0041"),("生年月日","1985年4月12日"),("会員番号","1234 5678 9012")]
    y = 90 - scroll
    for k,v in rows:
        d.text((60,y),k,font=font,fill=(90,90,90)); d.rectangle([260,y-6,900,y+36],outline=(180,180,180),fill="white")
        d.text((272,y),v,font=font,fill=(20,20,20)); y += 62
    d.text((60,y+10),"メモ: 佐藤花子様より 03-5555-1234 に折り返し希望。お客様対応は丁寧に。",font=small,fill=(20,20,20))
    d.text((60,y+50),"操作手順: 「保存」ボタンを押して確定します（2026-11-14 更新）",font=small,fill=(20,20,20))
    d.text((960,100),"検索: "+typed,font=small,fill=(20,20,20))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
cv2.imwrite("test/sample.png", screen())
fps=30; w=cv2.VideoWriter("test/sample_noaudio.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280,720))
for i in range(fps*6):
    t=i/fps; scroll=int(max(0,min(1,(t-2)/2))*120); typed="suzuki@test.jp"[:int(max(0,t-1)*5)]
    w.write(screen(scroll, typed))
w.release()
