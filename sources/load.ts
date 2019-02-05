import * as CSVParse from 'csv-parse/lib/sync'
import * as Fs from 'fs'

export const load = (file: string, training: boolean): void => {
  if (training) {
    return CSVParse(Fs.readFileSync(file).toString())
      .slice(1)
      .map(
        (image: any): any => ({
          label: parseInt(image[0]),
          data: image.slice(1).map((num: string): number => parseInt(num))
        })
      )
  }
  return CSVParse(Fs.readFileSync(file).toString()).map(
    (image: any): any => image.map((num: string): number => parseInt(num))
  )
}
