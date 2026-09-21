export const SIZE = 58;
export const round = (value: number) => Number(value.toFixed(3));
// Pythonの整数座標を、タイル・コマ共通のSVG座標に変換する。
export const boardPoint = (x: number, y: number) => ({
  x: round(400 + x * SIZE / 2), y: round(333 + y * Math.sqrt(3) * SIZE / 2),
});
