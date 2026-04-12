'use client'

import { create } from 'zustand'
import type { TextBlockInput } from '@/lib/api/schemas'

const MAX_HISTORY = 30

type TextBlockHistoryState = {
  // Per-document undo/redo stacks — each entry is a full snapshot of all
  // blocks BEFORE the operation so popping it restores that prior state.
  undoStacks: Record<string, TextBlockInput[][]>
  redoStacks: Record<string, TextBlockInput[][]>
  // Push the current state onto the undo stack before a mutating operation.
  // Clears the redo stack for that document.
  push: (docId: string, snapshot: TextBlockInput[]) => void
  popUndo: (docId: string) => TextBlockInput[] | undefined
  pushRedo: (docId: string, snapshot: TextBlockInput[]) => void
  popRedo: (docId: string) => TextBlockInput[] | undefined
  canUndo: (docId: string) => boolean
  canRedo: (docId: string) => boolean
  clear: (docId: string) => void
}

export const useTextBlockHistoryStore = create<TextBlockHistoryState>(
  (set, get) => ({
    undoStacks: {},
    redoStacks: {},

    push: (docId, snapshot) =>
      set((state) => {
        const prev = state.undoStacks[docId] ?? []
        const next = [...prev, snapshot].slice(-MAX_HISTORY)
        return {
          undoStacks: { ...state.undoStacks, [docId]: next },
          redoStacks: { ...state.redoStacks, [docId]: [] },
        }
      }),

    popUndo: (docId) => {
      const stack = get().undoStacks[docId] ?? []
      if (stack.length === 0) return undefined
      const snapshot = stack[stack.length - 1]
      set((state) => ({
        undoStacks: {
          ...state.undoStacks,
          [docId]: (state.undoStacks[docId] ?? []).slice(0, -1),
        },
      }))
      return snapshot
    },

    pushRedo: (docId, snapshot) =>
      set((state) => ({
        redoStacks: {
          ...state.redoStacks,
          [docId]: [...(state.redoStacks[docId] ?? []), snapshot],
        },
      })),

    popRedo: (docId) => {
      const stack = get().redoStacks[docId] ?? []
      if (stack.length === 0) return undefined
      const snapshot = stack[stack.length - 1]
      set((state) => ({
        redoStacks: {
          ...state.redoStacks,
          [docId]: (state.redoStacks[docId] ?? []).slice(0, -1),
        },
      }))
      return snapshot
    },

    canUndo: (docId) => (get().undoStacks[docId]?.length ?? 0) > 0,
    canRedo: (docId) => (get().redoStacks[docId]?.length ?? 0) > 0,

    clear: (docId) =>
      set((state) => ({
        undoStacks: { ...state.undoStacks, [docId]: [] },
        redoStacks: { ...state.redoStacks, [docId]: [] },
      })),
  }),
)
