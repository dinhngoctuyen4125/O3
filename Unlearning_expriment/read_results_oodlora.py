import json
import pandas

SCALE=str(0.1)
writer = pandas.ExcelWriter("all_results_scale_new_ood_01.xlsx", engine='xlsxwriter')
sheet_name = "unlearn"
sheet = writer.book.add_worksheet(sheet_name)

row = 0 #
for SEED in [0]:
    result_file = f"./SCALE_{SCALE}_seed_{SEED}_o_unlearn_lora_checkpoints/test_noretain_C_seed{SEED}_oodlora_lora_forget_D_test.json"

    with open(result_file, 'r') as f:
        data = json.load(f)
    dep = data.get('num_deprecated', 0)
    rep = data.get('num_replacement', 0)
    mis = data.get('num_mismatch', 0)
    total = data.get('total', 0)

    print(f"SEED: {SEED} ***********************")
    print(f"  D_test: total={total}, deprecated={dep}, replacement={rep}, mismatch={mis}")

    sheet.write(row, 0, f"SEED: {SEED}")
    sheet.write(row, 1, f"D_test")
    sheet.write(row, 2, f"dep:{dep}")
    sheet.write(row, 3, f"rep:{rep}")
    sheet.write(row, 4, f"mis:{mis}")
    sheet.write(row, 5, f"total:{total}")

    row += 1

writer.close()

