import { del, get, postFor } from "@/lib/api"
import type {
  DatasetShape,
  DatasetsList,
  ItemDetail,
  ItemsPage,
  NewDataset,
  RecordMeta,
  RecordResult,
  VerifyResult,
} from "@/lib/datasets/types"

const BASE = "/api/datasets"

const qs = (params: Record<string, string | number | undefined>) =>
  new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== undefined && v !== "") as [string, string][],
  ).toString()

export const datasetUrls = {
  audio: (ds: string, id: string) => `${BASE}/audio?${qs({ ds, id })}`,
}

export const datasetsApi = {
  list: () => get<DatasetsList>(BASE),
  create: (name: string, languagesRaw: string) =>
    postFor<NewDataset>(`${BASE}?${qs({ name, languages: languagesRaw })}`),
  shape: (ds: string) => get<DatasetShape>(`${BASE}/shape?${qs({ ds })}`),
  items: (ds: string, params: { offset: number; q: string; order?: string }) =>
    get<ItemsPage>(`${BASE}/items?${qs({ ds, ...params })}`),
  item: (ds: string, id: string) => get<ItemDetail>(`${BASE}/item?${qs({ ds, id })}`),
  verify: (ds: string) => get<VerifyResult>(`${BASE}/verify?${qs({ ds })}`),
  deleteAudio: (ds: string, id: string) => del<{ deleted_audio: string }>(`${BASE}/audio?${qs({ ds, id })}`),
  deleteItem: (ds: string, id: string) => del<{ deleted: string }>(`${BASE}/item?${qs({ ds, id })}`),
  record: (ds: string, pcm: Int16Array, meta: RecordMeta) =>
    postFor<RecordResult>(
      `${BASE}/record?${qs({
        ds,
        prefix: meta.prefix,
        speakers: meta.speakers,
        src_lang: meta.srcLang,
        speaker: meta.speaker,
      })}`,
      { headers: { "Content-Type": "application/octet-stream" }, body: pcm.buffer as ArrayBuffer },
    ),
}
