'use client'

import { useRef, useState, useMemo, useCallback, useEffect } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import { CheckIcon, ChevronDownIcon } from 'lucide-react'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'
import { fetchGoogleFont } from '@/lib/api/system/system'

const ITEM_HEIGHT = 28
const MAX_VISIBLE = 10

type FontOption = {
  familyName: string
  postScriptName: string
  source: 'custom' | 'system' | 'google'
  category?: string | null
  cached: boolean
}

type FontLoadState = 'idle' | 'loading' | 'ready' | 'error'

const MAX_CONCURRENT_FONT_LOADS = 8
const fontLoadStateCache = new Map<string, FontLoadState>()
const fontLoadPromises = new Map<string, Promise<void>>()
const fontLoadQueue: Array<() => void> = []
let activeFontLoads = 0

const fontKey = (source: string, family: string, postScriptName: string) =>
  source === 'custom' ? `${source}:${postScriptName}` : `${source}:${family}`

const scheduleFontLoad = (task: () => Promise<void>): Promise<void> =>
  new Promise((resolve, reject) => {
    const start = () => {
      activeFontLoads += 1
      task()
        .then(resolve)
        .catch(reject)
        .finally(() => {
          activeFontLoads -= 1
          const next = fontLoadQueue.shift()
          if (next) next()
        })
    }

    if (activeFontLoads < MAX_CONCURRENT_FONT_LOADS) {
      start()
      return
    }

    fontLoadQueue.push(start)
  })

const loadFontPreview = (
  family: string,
  postScriptName: string,
  source: string,
): Promise<void> => {
  if (source === 'system') return Promise.resolve()
  if (source !== 'google') return Promise.resolve()

  const key = fontKey(source, family, postScriptName)
  const existing = fontLoadPromises.get(key)
  if (existing) return existing
  if (fontLoadStateCache.get(key) === 'ready') return Promise.resolve()

  fontLoadStateCache.set(key, 'loading')

  const promise = scheduleFontLoad(async () => {
    if (source === 'google') {
      await fetchGoogleFont(encodeURIComponent(family))
    }

    const url = `/api/v1/fonts/google/${encodeURIComponent(family)}/file`
    const face = new FontFace(family, `url(${url})`)
    const loaded = await face.load()
    document.fonts.add(loaded)
    fontLoadStateCache.set(key, 'ready')
  })
    .catch((error) => {
      fontLoadStateCache.set(key, 'error')
      throw error
    })
    .finally(() => {
      fontLoadPromises.delete(key)
    })

  fontLoadPromises.set(key, promise)
  return promise
}

type FontSelectProps = {
  value: string
  options: FontOption[]
  disabled?: boolean
  placeholder?: string
  className?: string
  triggerClassName?: string
  triggerStyle?: React.CSSProperties
  onChange: (value: string) => void
  'data-testid'?: string
}

function useGoogleFontPreview(
  family: string,
  postScriptName: string,
  source: string,
  isVisible: boolean,
) {
  const key = fontKey(source, family, postScriptName)
  const [state, setState] = useState<FontLoadState>(() => {
    if (source === 'system') return 'ready'
    if (source !== 'google') return 'ready'
    return fontLoadStateCache.get(key) ?? 'idle'
  })

  useEffect(() => {
    if (source === 'system') {
      setState('ready')
      return
    }
    if (source !== 'google') {
      setState('ready')
      return
    }
    setState(fontLoadStateCache.get(key) ?? 'idle')
  }, [key, source])

  useEffect(() => {
    if (!isVisible || state !== 'idle') return
    if (source === 'system') return

    let cancelled = false
    setState('loading')

    loadFontPreview(family, postScriptName, source)
      .then(() => {
        if (!cancelled) setState('ready')
      })
      .catch(() => {
        if (!cancelled) setState('error')
      })

    return () => {
      cancelled = true
    }
  }, [family, postScriptName, source, isVisible, state])

  return state
}

function FontRow({
  font,
  selected,
  style,
  isVisible,
  onClick,
}: {
  font: FontOption
  selected: boolean
  style: React.CSSProperties
  isVisible: boolean
  onClick: () => void
}) {
  const loadState = useGoogleFontPreview(
    font.familyName,
    font.postScriptName,
    font.source,
    isVisible,
  )

  return (
    <button
      type='button'
      className={cn(
        'hover:bg-accent hover:text-accent-foreground absolute left-0 flex w-full cursor-default items-center gap-1.5 rounded-sm px-2 text-xs select-none',
        selected && 'bg-accent',
      )}
      style={{
        ...style,
        fontFamily: loadState === 'ready' ? font.familyName : undefined,
      }}
      onClick={onClick}
    >
      <span className='flex size-3 shrink-0 items-center justify-center'>
        {selected && <CheckIcon className='size-3' />}
      </span>
      <span className='truncate'>{font.familyName}</span>
      {font.source === 'google' && (
        <span className='text-muted-foreground ml-auto shrink-0 text-[9px]'>
          {loadState === 'loading' ? '...' : 'G'}
        </span>
      )}
    </button>
  )
}

export function FontSelect({
  value,
  options,
  disabled,
  placeholder,
  className,
  triggerClassName,
  triggerStyle,
  onChange,
  ...props
}: FontSelectProps) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const filtered = useMemo(() => {
    let result = options
    if (search) {
      const lower = search.toLowerCase()
      result = result.filter((f) => f.familyName.toLowerCase().includes(lower))
    }
    return result
  }, [options, search])

  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ITEM_HEIGHT,
    overscan: 5,
    enabled: open,
  })

  const viewportRef = useCallback(
    (node: HTMLDivElement | null) => {
      scrollRef.current = node
      if (node) virtualizer.measure()
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [open],
  )

  const selectedLabel = options.find(
    (f) => f.postScriptName === value || f.familyName === value,
  )?.familyName

  const listHeight = Math.min(filtered.length, MAX_VISIBLE) * ITEM_HEIGHT

  const preloadFonts = useMemo(() => {
    if (!open) return []

    const selected = filtered.find(
      (font) => font.postScriptName === value || font.familyName === value,
    )
    const firstVisible = filtered.slice(0, 40)
    return selected
      ? [selected, ...firstVisible.filter((font) => font !== selected)]
      : firstVisible
  }, [filtered, open, value])

  useEffect(() => {
    if (!open) return

    preloadFonts.forEach((font, index) => {
      const run = () => {
        void loadFontPreview(
          font.familyName,
          font.postScriptName,
          font.source,
        ).catch(() => undefined)
      }

      if (index < 3) {
        run()
        return
      }

      if (typeof window !== 'undefined' && 'requestIdleCallback' in window) {
        ;(
          window as Window & { requestIdleCallback: (cb: () => void) => void }
        ).requestIdleCallback(run)
        return
      }

      setTimeout(run, 0)
    })
  }, [open, preloadFonts])

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next)
        if (!next) setSearch('')
      }}
    >
      <PopoverTrigger
        disabled={disabled}
        data-testid={props['data-testid']}
        className={cn(
          "border-input data-[placeholder]:text-muted-foreground [&_svg:not([class*='text-'])]:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 dark:bg-input/30 dark:hover:bg-input/50 flex h-7 w-full items-center justify-between gap-1.5 rounded-md border bg-transparent px-2 py-1 text-xs whitespace-nowrap shadow-xs transition-[color,box-shadow] outline-none focus-visible:ring-[3px] disabled:cursor-not-allowed disabled:opacity-50",
          triggerClassName,
        )}
        style={triggerStyle}
      >
        <span className='truncate'>{selectedLabel ?? placeholder ?? ''}</span>
        <ChevronDownIcon className='size-3.5 shrink-0 opacity-50' />
      </PopoverTrigger>
      <PopoverContent
        className={cn('w-(--radix-popover-trigger-width) p-0', className)}
        align='start'
        onOpenAutoFocus={(e) => {
          e.preventDefault()
          inputRef.current?.focus()
        }}
      >
        <input
          ref={inputRef}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder='Search fonts…'
          className='placeholder:text-muted-foreground w-full border-b bg-transparent px-2 py-1.5 text-xs outline-none'
        />
        <ScrollArea
          className='relative'
          style={{ height: listHeight }}
          viewportRef={viewportRef}
        >
          <div
            style={{
              height: virtualizer.getTotalSize(),
              position: 'relative',
            }}
          >
            {virtualizer.getVirtualItems().map((vi) => {
              const font = filtered[vi.index]
              const selected =
                font.postScriptName === value || font.familyName === value
              return (
                <FontRow
                  key={vi.key}
                  font={font}
                  selected={selected}
                  style={{ height: ITEM_HEIGHT, top: vi.start }}
                  isVisible={true}
                  onClick={() => {
                    onChange(font.postScriptName)
                    setOpen(false)
                    setSearch('')
                  }}
                />
              )
            })}
          </div>
        </ScrollArea>
        {filtered.length === 0 && (
          <div className='text-muted-foreground px-2 py-4 text-center text-xs'>
            No fonts found
          </div>
        )}
      </PopoverContent>
    </Popover>
  )
}
