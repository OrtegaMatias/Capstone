import { useCallback, useState } from 'react';

const STORAGE_KEY = 'capstone_dataset_id';

export function useDatasetId() {
  const [datasetId, setDatasetIdState] = useState<string | null>(() => localStorage.getItem(STORAGE_KEY));

  const setDatasetId = useCallback((value: string | null) => {
    if (value) {
      localStorage.setItem(STORAGE_KEY, value);
      setDatasetIdState(value);
    } else {
      localStorage.removeItem(STORAGE_KEY);
      setDatasetIdState(null);
    }
  }, []);

  return { datasetId, setDatasetId };
}
