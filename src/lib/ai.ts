import type {
  Document,
  Expedition,
  Station,
} from '@/lib/hooks';

const POLAR_RAG_ENDPOINT =
  import.meta.env.VITE_POLAR_RAG_ENDPOINT ||
  'http://127.0.0.1:8000';

export interface AISource {
  title: string;
  type: string;
  route: string;
  documentSlug?: string;
}

export interface AIResponse {
  answer: string;
  sources: AISource[];
}

function cleanText(value: unknown): string {
  if (
    value === null ||
    value === undefined
  ) {
    return '';
  }

  return String(value).trim();
}

function getExpeditionName(
  expedition: Expedition,
): string {
  const record = expedition as Expedition & {
    title?: string | null;
  };

  return (
    cleanText(record.name) ||
    cleanText(record.title) ||
    cleanText(record.slug) ||
    'Unnamed expedition'
  );
}

function getStationName(
  station: Station,
): string {
  return (
    cleanText(station.name) ||
    cleanText(station.slug) ||
    'Unknown station'
  );
}

function getDocumentName(
  document: Document,
): string {
  return (
    cleanText(document.title) ||
    cleanText(document.name) ||
    cleanText(document.slug) ||
    'Untitled document'
  );
}

function createStationSource(
  station: Station,
): AISource {
  const slug = cleanText(station.slug);

  return {
    title: getStationName(station),
    type: 'Station',
    route: slug
      ? `/explore/${slug}`
      : '/explore',
  };
}

function createExpeditionSource(
  expedition: Expedition,
): AISource {
  const slug = cleanText(expedition.slug);

  return {
    title: getExpeditionName(expedition),
    type: 'Expedition',
    route: slug
      ? `/expeditions/${slug}`
      : '/expeditions',
  };
}

function createDocumentSource(
  document: Document,
): AISource {
  const slug = cleanText(document.slug);

  return {
    title: getDocumentName(document),
    type:
      cleanText(document.type) ||
      'Document',
    route: slug
      ? `/knowledge/${slug}`
      : '/knowledge',
    documentSlug:
      slug || undefined,
  };
}

function normalizeSource(
  source: any,
  documents: Document[],
  expeditions: Expedition[],
  stations: Station[],
): AISource | null {
  if (!source) {
    return null;
  }

  const title = cleanText(
    source.title ||
    source.name ||
    source.document_title ||
    source.documentName,
  );

  const slug = cleanText(
    source.slug ||
    source.documentSlug ||
    source.document_slug,
  );

  const type = cleanText(
    source.type ||
    source.source_type ||
    source.category,
  );

  let route = cleanText(
    source.route,
  );

  if (route.startsWith('/stations/')) {
    route = route.replace(
      '/stations/',
      '/explore/',
    );
  }

  if (!route && slug) {
    const station = stations.find(
      (item) =>
        cleanText(item.slug).toLowerCase() ===
        slug.toLowerCase(),
    );

    if (station) {
      route = `/explore/${station.slug}`;
    }

    const expedition =
      expeditions.find(
        (item) =>
          cleanText(
            item.slug,
          ).toLowerCase() ===
          slug.toLowerCase(),
      );

    if (!route && expedition) {
      route = `/expeditions/${expedition.slug}`;
    }

    const document =
      documents.find(
        (item) =>
          cleanText(
            item.slug,
          ).toLowerCase() ===
          slug.toLowerCase(),
      );

    if (!route && document) {
      route = `/knowledge/${document.slug}`;
    }
  }

  if (!route) {
    if (
      type
        .toLowerCase()
        .includes('station')
    ) {
      const station =
        stations.find(
          (item) =>
            cleanText(
              item.name,
            ).toLowerCase() ===
            title.toLowerCase() ||
            cleanText(
              item.slug,
            ).toLowerCase() ===
            title.toLowerCase(),
        );

      if (station) {
        route = `/explore/${station.slug}`;
      }
    }

    if (
      !route &&
      type
        .toLowerCase()
        .includes('expedition')
    ) {
      const expedition =
        expeditions.find(
          (item) =>
            getExpeditionName(
              item,
            ).toLowerCase() ===
            title.toLowerCase() ||
            cleanText(
              item.slug,
            ).toLowerCase() ===
            title.toLowerCase(),
        );

      if (expedition) {
        route = `/expeditions/${expedition.slug}`;
      }
    }

    if (!route) {
      const document =
        documents.find(
          (item) =>
            getDocumentName(
              item,
            ).toLowerCase() ===
            title.toLowerCase() ||
            cleanText(
              item.slug,
            ).toLowerCase() ===
            title.toLowerCase(),
        );

      if (document) {
        route = `/knowledge/${document.slug}`;
      }
    }
  }

  if (!title && !route) {
    return null;
  }

  return {
    title: title || 'Source',
    type: type || 'Source',
    route: route || '#',
    documentSlug:
      slug || undefined,
  };
}

function normalizeSources(
  sources: any[],
  documents: Document[],
  expeditions: Expedition[],
  stations: Station[],
): AISource[] {
  return sources
    .map((source) =>
      normalizeSource(
        source,
        documents,
        expeditions,
        stations,
      ),
    )
    .filter(Boolean) as AISource[];
}

function findStation(
  query: string,
  stations: Station[],
): Station | null {
  const normalizedQuery =
    query.toLowerCase();

  return (
    stations.find((station) => {
      const name = cleanText(
        station.name,
      ).toLowerCase();

      const slug = cleanText(
        station.slug,
      ).toLowerCase();

      return (
        normalizedQuery.includes(name) ||
        normalizedQuery.includes(slug) ||
        name.includes(normalizedQuery)
      );
    }) || null
  );
}

function findExpedition(
  query: string,
  expeditions: Expedition[],
): Expedition | null {
  const normalizedQuery =
    query.toLowerCase();

  return (
    expeditions.find((expedition) => {
      const name =
        getExpeditionName(
          expedition,
        ).toLowerCase();

      const slug = cleanText(
        expedition.slug,
      ).toLowerCase();

      return (
        normalizedQuery.includes(name) ||
        normalizedQuery.includes(slug) ||
        name.includes(normalizedQuery)
      );
    }) || null
  );
}

function stationExpeditionResponse(
  query: string,
  stations: Station[],
  expeditions: Expedition[],
): AIResponse | null {
  const station = findStation(
    query,
    stations,
  );

  if (!station) {
    return null;
  }

  const relatedExpeditions =
    expeditions.filter(
      (expedition) => {
        const stationId =
          cleanText(
            expedition.station_id,
          );

        return (
          stationId ===
          cleanText(station.id) ||
          cleanText(
            expedition.station?.id,
          ) === cleanText(station.id) ||
          cleanText(
            expedition.station?.slug,
          ) === cleanText(station.slug)
        );
      },
    );

  const normalizedQuery =
    query.toLowerCase();

  const asksForExpeditions =
    normalizedQuery.includes(
      'expedition',
    ) ||
    normalizedQuery.includes(
      'expeditions',
    ) ||
    normalizedQuery.includes(
      'associated',
    ) ||
    normalizedQuery.includes(
      'conducted',
    );

  if (!asksForExpeditions) {
    return null;
  }

  const expeditionNames =
    relatedExpeditions
      .map((expedition) => {
        const record =
          expedition as Expedition & {
            title?: string | null;
          };

        return (
          record.name ||
          record.title ||
          record.slug ||
          'Unnamed expedition'
        );
      })
      .filter(Boolean)
      .join(', ');

  const sources: AISource[] = [
    createStationSource(station),
    ...relatedExpeditions.map(
      createExpeditionSource,
    ),
  ];

  if (
    relatedExpeditions.length === 0
  ) {
    return {
      answer: `${getStationName(
        station,
      )} does not currently have any expeditions associated with it in the POLAR repository.`,
      sources: [
        createStationSource(station),
      ],
    };
  }

  return {
    answer: `${getStationName(
      station,
    )} is associated with the following expeditions in the POLAR repository: ${expeditionNames}.`,
    sources,
  };
}

function localFallback(
  query: string,
  documents: Document[],
  expeditions: Expedition[],
  stations: Station[],
): AIResponse {
  const station = findStation(
    query,
    stations,
  );

  if (station) {
    const lowerQuery =
      query.toLowerCase();

    if (
      lowerQuery.includes('research') ||
      lowerQuery.includes('focus') ||
      lowerQuery.includes('study')
    ) {
      const researchFocus =
        Array.isArray(
          station.research_focus,
        )
          ? station.research_focus.join(
            ', ',
          )
          : cleanText(
            station.research_focus,
          );

      return {
        answer: `### ${getStationName(
          station,
        )}

**Region:** ${cleanText(station.region) ||
          'Polar region'
          }

**Location:** ${cleanText(station.location) ||
          'Not specified'
          }

**Established:** ${station.established_year ||
          'Not specified'
          }

**Research focus:** ${researchFocus ||
          'Not specified'
          }

${cleanText(
            station.overview ||
            station.description,
          )}`,
        sources: [
          createStationSource(station),
        ],
      };
    }

    return {
      answer: `### ${getStationName(
        station,
      )}

**Region:** ${cleanText(station.region) ||
        'Polar region'
        }

**Location:** ${cleanText(station.location) ||
        'Not specified'
        }

**Established:** ${station.established_year ||
        'Not specified'
        }

${cleanText(
          station.overview ||
          station.description,
        )}`,
      sources: [
        createStationSource(station),
      ],
    };
  }

  const expedition =
    findExpedition(
      query,
      expeditions,
    );

  if (expedition) {
    return {
      answer: `### ${getExpeditionName(
        expedition,
      )}

**Region:** ${cleanText(expedition.region) ||
        'Not specified'
        }

**Year:** ${expedition.year ||
        'Not specified'
        }

**Objectives:** ${cleanText(
          expedition.objectives,
        ) || 'Not specified'
        }

${cleanText(
          expedition.description,
        )}`,
      sources: [
        createExpeditionSource(
          expedition,
        ),
      ],
    };
  }

  const normalizedQuery =
    query.toLowerCase();

  const matchingDocuments =
    documents.filter(
      (document) => {
        const searchable = [
          getDocumentName(document),
          cleanText(
            document.description,
          ),
          cleanText(
            document.category,
          ),
          cleanText(
            document.region,
          ),
          cleanText(
            document.institution,
          ),
          ...(Array.isArray(
            document.research_areas,
          )
            ? document.research_areas
            : []),
        ]
          .join(' ')
          .toLowerCase();

        return normalizedQuery
          .split(/\s+/)
          .some(
            (word) =>
              word.length > 3 &&
              searchable.includes(word),
          );
      },
    );

  if (
    matchingDocuments.length > 0
  ) {
    const topDocuments =
      matchingDocuments.slice(0, 5);

    const sourceText =
      topDocuments
        .map(
          (document) =>
            `- **${getDocumentName(
              document,
            )}** — ${cleanText(
              document.description,
            ) ||
            'No description available.'
            }`,
        )
        .join('\n');

    return {
      answer: `### Relevant POLAR resources

${sourceText}`,
      sources:
        topDocuments.map(
          createDocumentSource,
        ),
    };
  }

  return {
    answer:
      "I couldn't find a matching resource in the current POLAR knowledge repository. Try asking about a station, expedition, research topic, or document.",
    sources: [],
  };
}

/* =========================================================
   EXISTING POLAR RAG
   ========================================================= */

export async function askPolarAI(
  query: string,
  documents: Document[],
  expeditions: Expedition[],
  stations: Station[],
): Promise<AIResponse> {
  const cleanedQuery =
    query.trim();

  if (!cleanedQuery) {
    return {
      answer:
        'Please enter a question about polar science.',
      sources: [],
    };
  }

  const relationshipResponse =
    stationExpeditionResponse(
      cleanedQuery,
      stations,
      expeditions,
    );

  if (relationshipResponse) {
    return relationshipResponse;
  }

  try {
    const endpoint =
      `${POLAR_RAG_ENDPOINT}/api/ask`;

    const response =
      await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type':
            'application/json',
        },
        body: JSON.stringify({
          query: cleanedQuery,
        }),
      });

    if (!response.ok) {
      throw new Error(
        `POLAR RAG returned ${response.status}`,
      );
    }

    const data =
      await response.json();

    const answer =
      cleanText(data.answer) ||
      cleanText(data.response) ||
      cleanText(data.message);

    const backendSources =
      Array.isArray(
        data.citations,
      )
        ? data.citations
        : Array.isArray(
          data.sources,
        )
          ? data.sources
          : [];

    const sources =
      normalizeSources(
        backendSources,
        documents,
        expeditions,
        stations,
      );

    if (answer) {
      return {
        answer,
        sources,
      };
    }
  } catch (error) {
    console.warn(
      'POLAR RAG backend unavailable. Using local fallback.',
      error,
    );
  }

  return localFallback(
    cleanedQuery,
    documents,
    expeditions,
    stations,
  );
}

/* =========================================================
   POLAR OUTREACH CONTENT GENERATOR
   ========================================================= */

export interface PolarContentResponse {
  query: string;
  platform: string;
  content: string;
  citations: Array<{
    title: string;
    source: string;
    route: string;
    score?: number;
  }>;
}

export async function generatePolarContent(
  query: string,
  platform: string = 'website_article',
): Promise<PolarContentResponse> {
  const cleanedQuery =
    query.trim();

  if (!cleanedQuery) {
    throw new Error(
      'Please enter a topic for content generation.',
    );
  }

  const endpoint =
    `${POLAR_RAG_ENDPOINT}/api/generate-content`;

  const response =
    await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type':
          'application/json',
      },
      body: JSON.stringify({
        query: cleanedQuery,
        platform,
      }),
    });

  if (!response.ok) {
    let message =
      `POLAR content generation failed: ${response.status}`;

    try {
      const errorData =
        await response.json();

      if (
        errorData?.detail
      ) {
        message =
          typeof errorData.detail ===
            'string'
            ? errorData.detail
            : message;
      }
    } catch {
      // Keep default error.
    }

    throw new Error(message);
  }

  const data =
    await response.json();

  return {
    query:
      cleanText(data.query) ||
      cleanedQuery,

    platform:
      cleanText(data.platform) ||
      platform,

    content:
      cleanText(data.content) ||
      '',

    citations:
      Array.isArray(
        data.citations,
      )
        ? data.citations.map(
          (citation: any) => ({
            title:
              cleanText(
                citation.title,
              ) || 'Source',

            source:
              cleanText(
                citation.source,
              ) || 'POLAR',

            route:
              cleanText(
                citation.route,
              ) || '#',

            score:
              typeof citation.score ===
                'number'
                ? citation.score
                : undefined,
          }),
        )
        : [],
  };
}